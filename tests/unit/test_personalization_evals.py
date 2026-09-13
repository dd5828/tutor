"""个性化评测运行器单测（假 LLM / 真读写工具 + tmp 画像目录，不触网）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from nnnu.capabilities.math.capability import MathTutorCapability
from nnnu.runtime.registry import ToolRegistry
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageTracker
from nnnu.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from tests.evals.run_personalization_evals import (
    PersonaCase,
    PersonaTurn,
    PersonaTurnScore,
    build_score_prompt,
    drive_role,
    load_personalization_cases,
    run,
    score_turn,
    summarize,
)
from tests.unit.test_math_capability import (
    GIVE_UP_PATH,
    ScriptedLLM,
    ScriptedResponse,
    StubTracker,
)


def _write_cases(tmp_path: Any, lines: list[dict[str, Any]]) -> Any:
    path = tmp_path / "personalization.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def _min_case(**overrides: Any) -> dict[str, Any]:
    case = {
        "id": "persona-x-01",
        "role": "基础薄弱的学生",
        "session_id": "eval-persona-x-01",
        "expected_level": "基础",
        "turns": [{"question": "题一", "stuck_point": "卡点一", "replies": []}],
    }
    case.update(overrides)
    return case


def _persona() -> PersonaCase:
    return PersonaCase(
        id="persona-x-01",
        role="基础薄弱的学生",
        session_id="eval-persona-x-01",
        expected_level="基础",
        turns=[PersonaTurn(question="题一", stuck_point="卡点一", replies=[])],
    )


def test_load_valid_persona_case(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case()])
    cases = load_personalization_cases(path)
    assert cases == [_persona()]


def test_load_rejects_missing_turns(tmp_path: Any) -> None:
    raw = _min_case()
    raw.pop("turns")
    path = _write_cases(tmp_path, [raw])
    with pytest.raises(ValueError, match="turns"):
        load_personalization_cases(path)


def test_load_rejects_turn_missing_question(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case(turns=[{"stuck_point": "卡"}])])
    with pytest.raises(ValueError, match="question"):
        load_personalization_cases(path)


def test_load_rejects_bad_expected_level(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case(expected_level="超难")])
    with pytest.raises(ValueError, match="expected_level"):
        load_personalization_cases(path)


async def test_score_turn_first_turn_exempt() -> None:
    score = await score_turn(
        cast(LLMClient, ScriptedLLM([])), _persona(), 1, "对话", "摘要"
    )
    assert score.passed is None
    assert score.memory_referenced is None and score.level_matched is None


class _JudgeLLM:
    """返回预置判分 JSON 的假判分 LLM，并记录调用。"""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        self.calls.append({"messages": messages, **kwargs})
        yield self._payload


async def test_score_turn_valid_json_passes() -> None:
    judge = _JudgeLLM('{"memory_referenced": true, "level_matched": true, "notes": "点名了"}')
    score = await score_turn(cast(LLMClient, judge), _persona(), 2, "对话", "摘要")
    assert score.passed is True
    assert score.memory_referenced and score.level_matched
    # 判分低温
    assert judge.calls[0]["temperature"] == 0.0


async def test_score_turn_parse_failure_conservative_fail() -> None:
    judge = _JudgeLLM("抱歉我没法评分")
    score = await score_turn(cast(LLMClient, judge), _persona(), 2, "对话", "摘要")
    assert score.passed is False
    assert score.notes == "判分解析失败"


def test_score_prompt_contains_role_profile_and_rubric() -> None:
    prompt = build_score_prompt(_persona(), 3, "对话全文…", "画像摘要…")
    assert "基础薄弱的学生" in prompt
    assert "基础" in prompt  # 期望档位入标准
    assert "画像摘要…" in prompt
    assert "memory_referenced" in prompt and "level_matched" in prompt
    assert "上次你在" in prompt  # 引用判定的句式样例


def test_summarize_math() -> None:
    scores = [
        PersonaTurnScore("a", 1, None, None, None, "首轮"),
        PersonaTurnScore("a", 2, True, True, True, ""),
        PersonaTurnScore("a", 3, False, True, False, ""),
    ]
    assert summarize(scores) == (2, 1, 2)
    assert summarize([]) == (0, 0, 0)


def _capability_with_memory(llm: ScriptedLLM, memory_dir: Any) -> MathTutorCapability:
    registry = ToolRegistry()
    registry.register(ReadMemoryTool(memory_dir=memory_dir))
    registry.register(WriteMemoryTool(memory_dir=memory_dir))
    return MathTutorCapability(
        llm=cast(LLMClient, llm), usage=cast(UsageTracker, StubTracker()), tools=registry
    )


async def test_drive_role_two_turns_writes_and_reads_profile(tmp_path: Any) -> None:
    """真 Read/WriteMemoryTool + 假能力 LLM：画像逐轮累积、第二轮摘要含首轮。"""
    case = PersonaCase(
        id="persona-x-01",
        role="r",
        session_id="eval-persona-x-01",
        expected_level="常规",
        turns=[
            PersonaTurn(question="题一", stuck_point="卡点一", replies=["直接讲吧"]),
            PersonaTurn(question="题二", stuck_point="卡点二", replies=["直接讲吧"]),
        ],
    )
    # 每轮 6 次 LLM 调用（GIVE_UP 流程），共 12 响应
    llm = ScriptedLLM([ScriptedResponse(["x"]) for _ in range(len(GIVE_UP_PATH) * 2)])
    cap = _capability_with_memory(llm, tmp_path)
    rounds = await drive_role(cap, case, tmp_path)
    assert len(rounds) == 2
    # 第 1 轮开始前无画像；第 2 轮开始前画像已含首轮
    assert rounds[0][2] == ""
    assert "1 轮" in rounds[1][2]
    # 档案落盘且累计 2 轮
    data = json.loads((tmp_path / "eval-persona-x-01.json").read_text(encoding="utf-8"))
    assert data["turn_count"] == 2


async def test_run_limit_and_out_with_scripted_llms(tmp_path: Any) -> None:
    """run() 全流程（假能力 LLM + 假判分 LLM）：limit 截断 + 结果落盘。"""
    cases_path = _write_cases(tmp_path, [_min_case(), _min_case(id="persona-x-02")])
    cap_llm = ScriptedLLM([ScriptedResponse(["x"]) for _ in range(len(GIVE_UP_PATH) * 2)])
    judge_llm = ScriptedLLM(
        [
            ScriptedResponse(
                ['{"memory_referenced": true, "level_matched": true, "notes": "ok"}']
            )
        ]
    )
    out_path = tmp_path / "out.jsonl"
    scores = await run(
        cases_path,
        memory_dir=tmp_path / "mem",
        cap_llm=cast(LLMClient, cap_llm),
        judge_llm=cast(LLMClient, judge_llm),
        limit=1,
        out_path=out_path,
    )
    assert len(scores) == 1  # limit=1：只跑第一个角色的 1 轮
    assert scores[0].passed is None  # 首轮豁免
    written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["case_id"] == "persona-x-01"
