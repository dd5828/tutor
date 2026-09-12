"""评测运行器逻辑单测（假 LLM 驱动，不触网）。

覆盖：cases.jsonl 加载与校验、一轮对话驱动与对话还原、LLM 判分解析
（含保守降级与通过线）。真 LLM 的基线运行在 CI 之外手动执行。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from nnnu.capabilities.math.capability import MathTutorCapability
from nnnu.core import ChatError, StreamEvent, StreamEventType
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageTracker
from tests.evals.run_evals import (
    EvalCase,
    build_transcript,
    load_cases,
    run_case,
    score_case,
)


@dataclass
class ScriptedResponse:
    """一次预编排的 LLM 响应：文本块 + 可选异常。"""

    chunks: list[str]
    error: Exception | None = None


class ScriptedLLM:
    """按调用顺序返回预编排响应；多余调用直接断言失败。"""

    def __init__(self, responses: list[ScriptedResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls.append(
            {"messages": messages, "temperature": kwargs.get("temperature", 0.7)}
        )
        if not self._responses:
            raise AssertionError("意外的额外 LLM 调用")
        response = self._responses.pop(0)
        for chunk in response.chunks:
            yield chunk
        if response.error is not None:
            raise response.error


class StubTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


def _capability(llm: ScriptedLLM) -> MathTutorCapability:
    return MathTutorCapability(
        llm=cast(LLMClient, llm), usage=cast(UsageTracker, StubTracker())
    )


def _write_case(tmp_path: Path, obj: dict[str, Any]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---- cases.jsonl 加载与校验 ----

VALID_CASE: dict[str, Any] = {
    "id": "c-1",
    "level": "基础",
    "question": "q",
    "stuck_point": "s",
    "expected": ["变式题"],
}


def test_load_cases_parses_minimal_case(tmp_path: Path) -> None:
    cases = load_cases(_write_case(tmp_path, VALID_CASE))

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "c-1"
    assert case.level == "基础"
    assert case.replies == []  # 可选字段缺省为空


def test_load_cases_rejects_bad_level(tmp_path: Path) -> None:
    bad = dict(VALID_CASE, level="超难")
    with pytest.raises(ValueError, match="level"):
        load_cases(_write_case(tmp_path, bad))


def test_load_cases_rejects_unknown_expected_key(tmp_path: Path) -> None:
    bad = dict(VALID_CASE, expected=["直接报答案"])
    with pytest.raises(ValueError, match="未知期望行为"):
        load_cases(_write_case(tmp_path, bad))


def test_load_cases_rejects_missing_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="缺字段"):
        load_cases(_write_case(tmp_path, {"id": "c-1", "level": "基础"}))


# ---- 对话驱动与还原 ----


async def test_run_case_drives_dialogue_and_builds_transcript() -> None:
    llm = ScriptedLLM(
        [
            ScriptedResponse(["这题是求极限。"]),
            ScriptedResponse(['{"level": "基础"}']),
            ScriptedResponse(["先看它是什么未定式？"]),
            ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "solved"}']),
            ScriptedResponse(["考点总结。"]),
            ScriptedResponse(["变式题：…"]),
        ]
    )
    case = EvalCase(
        id="t-01",
        level="基础",
        question="求 lim(x→0) sinx/x",
        stuck_point="卡在第一步",
        replies=["答案是 1"],
        expected=["变式题"],
    )

    transcript, error = await run_case(_capability(llm), case)

    assert error is None
    assert "[学生] 卡在第一步" in transcript
    assert "[学生] 答案是 1" in transcript
    assert "[问] 你卡在哪一步？算到哪了？" in transcript
    assert "[AI] 先看它是什么未定式？" in transcript


async def test_run_case_captures_llm_error() -> None:
    llm = ScriptedLLM(
        [
            ScriptedResponse(["题意为…"]),
            ScriptedResponse([], error=ChatError("模型超时")),
        ]
    )
    case = EvalCase(
        id="t-02", level="基础", question="q", stuck_point="s", expected=["变式题"]
    )

    transcript, error = await run_case(_capability(llm), case)

    assert isinstance(error, ChatError)
    assert "[问]" in transcript  # 出错前已产出的部分仍在


async def test_build_transcript_formats_dialogue() -> None:
    events = [
        StreamEvent(type=StreamEventType.CONTENT, content="AI 第一句"),
        StreamEvent(type=StreamEventType.WAIT_FOR_INPUT, content="你的答案？"),
        StreamEvent(type=StreamEventType.CONTENT, content="AI 第二句"),
    ]

    transcript = build_transcript(events, ["答一", "答二"])

    assert transcript == "[AI] AI 第一句\n[问] 你的答案？\n[学生] 答一\n[AI] AI 第二句"


async def test_build_transcript_includes_stage_markers() -> None:
    events = [
        StreamEvent(type=StreamEventType.STAGE_START, stage="引导"),
        StreamEvent(type=StreamEventType.CONTENT, content="提示一句"),
    ]

    transcript = build_transcript(events, [])

    # 阶段标记是判分的关键上下文：讲解阶段的完整讲解不属"直接给答案"违规
    assert transcript == "〔进入阶段：引导〕\n[AI] 提示一句"


# ---- LLM 判分 ----

FULL_SCORES = (
    '{"one_step": 3, "error_layer": 3, "ladder": 2, '
    '"explain": 2, "variant": 2, "latex": 1}'
)


def _case() -> EvalCase:
    return EvalCase(
        id="c-1", level="基础", question="q", stuck_point="s", expected=["变式题"]
    )


async def test_score_case_parses_valid_json() -> None:
    llm = ScriptedLLM(
        [ScriptedResponse([f'{{"violations": [], "scores": {FULL_SCORES}, "notes": "好"}}'])]
    )
    score = await score_case(cast(LLMClient, llm), _case(), "对话全文")

    assert score.total == 13
    assert score.passed is True
    assert score.violations == []
    assert score.notes == "好"


async def test_score_case_below_threshold_fails() -> None:
    low = '{"one_step": 3, "error_layer": 2, "ladder": 1, "explain": 1, "variant": 0, "latex": 0}'
    llm = ScriptedLLM(
        [ScriptedResponse([f'{{"violations": [], "scores": {low}, "notes": "x"}}'])]
    )
    score = await score_case(cast(LLMClient, llm), _case(), "对话全文")

    assert score.total == 7
    assert score.passed is False  # 通过线：总分 ≥ 8


async def test_score_case_violation_fails_even_with_full_total() -> None:
    llm = ScriptedLLM(
        [
            ScriptedResponse(
                [
                    '{"violations": [{"rule": "直接给完整答案", "evidence": "..."}], '
                    f'"scores": {FULL_SCORES}, "notes": "x"}}'
                ]
            )
        ]
    )
    score = await score_case(cast(LLMClient, llm), _case(), "对话全文")

    assert score.total == 13
    assert score.passed is False  # 严重违规一票否决
    assert score.violations == ["直接给完整答案"]


async def test_score_case_malformed_fails_conservatively() -> None:
    llm = ScriptedLLM([ScriptedResponse(["抱歉无法评分"])])
    score = await score_case(cast(LLMClient, llm), _case(), "对话全文")

    assert score.total == 0
    assert score.passed is False
    assert score.violations == ["评分解析失败"]


async def test_score_prompt_contains_expected_and_rubric() -> None:
    llm = ScriptedLLM([ScriptedResponse(['{"violations": [], "scores": {}, "notes": ""}'])])
    case = _case()
    case.expected = ["错因定位=公式", "变式题"]

    await score_case(cast(LLMClient, llm), case, "对话全文")

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "错因定位=公式" in prompt
    assert "评分标准（满分 13）" in prompt
    assert llm.calls[0]["temperature"] == 0.0
