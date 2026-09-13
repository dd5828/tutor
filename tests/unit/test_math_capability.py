"""MathTutorCapability 单测（注入预编排假 LLMClient，不触网）。

_drive 并发驱动一轮：订阅者遇到 WAIT_FOR_INPUT 事件时按顺序投递学生回复，
模拟 orchestrator 反问桥的传输层回路。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from nnnu.capabilities.math.capability import MathTutorCapability, _collect_references
from nnnu.capabilities.math.pedagogy import (
    DEFAULT_LEVEL,
    Judgement,
    estimate_level,
    judge_answer,
)
from nnnu.capabilities.math.prompts import REFERENCES_PROMPT, SYSTEM_PROMPT
from nnnu.core import (
    BaseTool,
    ChatError,
    StreamBus,
    StreamEvent,
    StreamEventType,
    ToolResult,
    TurnContext,
)
from nnnu.runtime.registry import ToolRegistry
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo, UsageTracker

STAGE_START = StreamEventType.STAGE_START
STAGE_END = StreamEventType.STAGE_END
CONTENT = StreamEventType.CONTENT
WAIT = StreamEventType.WAIT_FOR_INPUT
SOURCES = StreamEventType.SOURCES


@dataclass
class ScriptedResponse:
    """一次预编排的 LLM 响应：文本块 + 可选用量尾块 + 可选异常。"""

    chunks: list[str]
    usage: UsageInfo | None = None
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
        on_usage = kwargs.get("on_usage")
        if response.usage is not None and callable(on_usage):
            on_usage(response.usage)
        if response.error is not None:
            raise response.error


class StubTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


def _capability(
    responses: list[ScriptedResponse],
    tools: ToolRegistry | None = None,
) -> tuple[MathTutorCapability, ScriptedLLM, StubTracker]:
    llm = ScriptedLLM(responses)
    tracker = StubTracker()
    # 默认注入全新空注册表（不落全局单例）：本文件测试与知识库/其他测试隔离
    cap = MathTutorCapability(
        llm=cast(LLMClient, llm),
        usage=cast(UsageTracker, tracker),
        tools=tools if tools is not None else ToolRegistry(),
    )
    return cap, llm, tracker


async def _drive(
    cap: MathTutorCapability,
    ctx: TurnContext,
    replies: list[str],
) -> tuple[list[StreamEvent], Exception | None]:
    """并发驱动一轮；返回完整事件流与 run 的异常（正常为 None）。"""
    bus = StreamBus()
    run_task = asyncio.create_task(cap.run(ctx, bus))

    async def consume() -> list[StreamEvent]:
        events: list[StreamEvent] = []
        async for event in bus.subscribe():
            events.append(event)
            if event.type == StreamEventType.WAIT_FOR_INPUT:
                bus.submit_input(replies.pop(0) if replies else "")
        return events

    consumer = asyncio.create_task(consume())
    error: Exception | None = None
    try:
        await run_task
    except Exception as exc:
        error = exc
    bus.mark_closed()
    events = await consumer
    return events, error


# 完整走一遍的预编排：诊断 → 估档位 → 引导两轮（对+继续、对+已解出）→ 讲解 → 巩固
HAPPY_PATH = [
    ScriptedResponse(["这题是求极限 ", "$\\lim_{x\\to0}\\frac{\\sin x}{x}$。", "你卡在哪？"]),
    ScriptedResponse(['{"level": "常规"}']),
    ScriptedResponse(["先判断它是哪种未定式？"]),
    ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "continue"}']),
    ScriptedResponse(["那 0/0 型通常用什么工具？"]),
    ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "solved"}']),
    ScriptedResponse(["考点：重要极限。", "你做对了，总结如下。"]),
    ScriptedResponse(["变式题：", "$\\lim_{x\\to0}\\frac{\\tan x}{x}$", "总结：…"]),
]


async def test_manifest_fields() -> None:
    manifest = MathTutorCapability.manifest
    assert manifest.name == "math"
    assert manifest.stages == ["诊断", "引导", "讲解", "巩固"]
    assert manifest.tools_used == ["rag", "check_answer", "question_bank"]


async def test_run_full_flow_stage_and_event_sequence() -> None:
    cap, llm, _ = _capability(HAPPY_PATH)
    ctx = TurnContext(user_message="求 lim(x→0) sin(x)/x")
    events, error = await _drive(cap, ctx, ["卡在第一步", "是 0/0 型吧", "用洛必达"])

    assert error is None
    assert [e.type for e in events] == [
        STAGE_START, CONTENT, CONTENT, CONTENT, WAIT, STAGE_END,  # 诊断
        STAGE_START, CONTENT, WAIT, CONTENT, WAIT, STAGE_END,  # 引导两轮
        STAGE_START, CONTENT, CONTENT, STAGE_END,  # 讲解
        STAGE_START, CONTENT, CONTENT, CONTENT, STAGE_END,  # 巩固
    ]
    assert [e.stage for e in events if e.type == STAGE_START] == [
        "诊断",
        "引导",
        "讲解",
        "巩固",
    ]
    assert ctx.metadata["level"] == "常规"
    # 消息组装：系统提示词打头；判答含学生回答；第二轮提示含第一轮过程
    assert llm.calls[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "0/0" in llm.calls[3]["messages"][0]["content"]
    # 流式调用组装 [system, user]：过程记录在 user 消息（下标 1）里
    assert "学生答：是 0/0 型吧" in llm.calls[4]["messages"][1]["content"]


async def test_run_structured_calls_use_zero_temperature() -> None:
    cap, llm, _ = _capability(HAPPY_PATH)
    _, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])
    assert error is None
    temps = [call["temperature"] for call in llm.calls]
    # 调用序：诊断(0) 估档位(1) 提示(2) 判答(3) 提示(4) 判答(5) 讲解(6) 巩固(7)
    assert temps[1] == 0.0 and temps[3] == 0.0 and temps[5] == 0.0
    assert all(t == 0.7 for i, t in enumerate(temps) if i not in (1, 3, 5))


# 两轮答错（错因=公式）→ 第三轮降级为更细提示 → 答对解出
WRONG_PATH = [
    ScriptedResponse(["题意为…"]),
    ScriptedResponse(['{"level": "基础"}']),
    ScriptedResponse(["提示一"]),
    ScriptedResponse(['{"correct": false, "error_layer": "公式", "next_step": "continue"}']),
    ScriptedResponse(["提示二"]),
    ScriptedResponse(['{"correct": false, "error_layer": "公式", "next_step": "continue"}']),
    ScriptedResponse(["提示三"]),
    ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "solved"}']),
    ScriptedResponse(["讲解内容"]),
    ScriptedResponse(["变式与总结"]),
]


async def test_run_wrong_answer_escalates_hint_level_and_explain_recalls_error() -> None:
    cap, llm, _ = _capability(WRONG_PATH)
    ctx = TurnContext(user_message="求导题")
    events, error = await _drive(cap, ctx, ["卡住", "答案1", "答案2", "答案3"])

    assert error is None
    # 连续两次答错后，第三轮提示升级为更细提示并携带错因（user 消息内）
    assert "更细提示" in llm.calls[6]["messages"][1]["content"]
    assert "公式" in llm.calls[6]["messages"][1]["content"]
    # 讲解阶段回扣错因
    assert "公式" in llm.calls[8]["messages"][1]["content"]
    # 引导阶段三轮反问 + 诊断一轮 = 4 次等待输入
    assert sum(1 for e in events if e.type == WAIT) == 4


GIVE_UP_PATH = [
    ScriptedResponse(["题意为…"]),
    ScriptedResponse(['{"level": "常规"}']),
    ScriptedResponse(["提示一"]),
    ScriptedResponse(['{"correct": false, "error_layer": "方法", "next_step": "explain"}']),
    ScriptedResponse(["完整讲解…"]),
    ScriptedResponse(["变式…"]),
]


async def test_run_judge_explain_stops_guidance() -> None:
    cap, llm, _ = _capability(GIVE_UP_PATH)
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "直接讲吧"])

    assert error is None
    assert len(llm.calls) == 6  # 判 explain 即收，无第二轮提示
    assert [e.stage for e in events if e.type == STAGE_START] == [
        "诊断",
        "引导",
        "讲解",
        "巩固",
    ]


async def test_run_empty_reply_goes_straight_to_explain() -> None:
    cap, llm, _ = _capability(
        [
            ScriptedResponse(["题意为…"]),
            ScriptedResponse(['{"level": "常规"}']),
            ScriptedResponse(["提示一"]),
            ScriptedResponse(["讲解…"]),
            ScriptedResponse(["变式…"]),
        ]
    )
    # 引导轮不投递回复（模拟超时/断连空输入）：不再判答，直接讲解
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡住"])

    assert error is None
    assert len(llm.calls) == 5
    assert sum(1 for e in events if e.type == WAIT) == 2


async def test_run_records_accumulated_usage_once() -> None:
    responses = [
        ScriptedResponse(["x"], UsageInfo(1, 2, 3)),
        ScriptedResponse(["y"], UsageInfo(4, 5, 9)),
        ScriptedResponse(["h"], UsageInfo(7, 8, 15)),
        ScriptedResponse(
            ['{"correct": true, "error_layer": "无", "next_step": "solved"}'],
            UsageInfo(10, 11, 21),
        ),
        ScriptedResponse(["e"], UsageInfo(13, 14, 27)),
        ScriptedResponse(["c"], UsageInfo(16, 17, 33)),
    ]
    cap, _, tracker = _capability(responses)
    ctx = TurnContext(session_id="s1", user_message="题", metadata={"turn_id": "t1"})

    _, error = await _drive(cap, ctx, ["卡", "答案"])
    assert error is None

    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record["session_id"] == "s1"
    assert record["turn_id"] == "t1"
    assert record["capability"] == "math"
    assert record["usage"] == UsageInfo(51, 57, 108)
    assert record["duration"] >= 0


async def test_run_llm_error_propagates_stage_pair_closes_and_usage_recorded() -> None:
    cap, _, tracker = _capability(
        [
            ScriptedResponse(["题意为…"], UsageInfo(1, 2, 3)),
            ScriptedResponse([], error=ChatError("模型超时")),
        ]
    )
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡住"])

    assert isinstance(error, ChatError)
    # 异常也不破坏 stage 配对：诊断 start/end 都在
    assert [e.type for e in events] == [STAGE_START, CONTENT, WAIT, STAGE_END]
    # finally 中仍记录已产生的用量
    assert len(tracker.records) == 1
    assert tracker.records[0]["usage"] == UsageInfo(1, 2, 3)


async def test_judge_answer_parses_json() -> None:
    llm = ScriptedLLM(
        [ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "solved"}'])]
    )
    judgement = await judge_answer(
        cast(LLMClient, llm), question="q", last_hint="h", reply="r"
    )
    assert judgement == Judgement(correct=True, error_layer="", next_step="solved")
    assert llm.calls[0]["temperature"] == 0.0


async def test_judge_answer_tolerates_code_fence() -> None:
    llm = ScriptedLLM(
        [
            ScriptedResponse(
                ['```json\n{"correct": false, "error_layer": "公式", "next_step": "continue"}\n```']
            )
        ]
    )
    judgement = await judge_answer(
        cast(LLMClient, llm), question="q", last_hint="h", reply="r"
    )
    assert judgement == Judgement(correct=False, error_layer="公式", next_step="continue")


async def test_judge_answer_malformed_falls_back_conservatively() -> None:
    llm = ScriptedLLM([ScriptedResponse(["抱歉我没法判断"])])
    judgement = await judge_answer(
        cast(LLMClient, llm), question="q", last_hint="h", reply="r"
    )
    # 保守降级：视为未解出、进入讲解
    assert judgement == Judgement(correct=False, error_layer="", next_step="explain")


async def test_judge_answer_invalid_enum_values_fall_back() -> None:
    llm = ScriptedLLM(
        [ScriptedResponse(['{"correct": true, "error_layer": "粗心", "next_step": "whatever"}'])]
    )
    judgement = await judge_answer(
        cast(LLMClient, llm), question="q", last_hint="h", reply="r"
    )
    assert judgement == Judgement(correct=True, error_layer="", next_step="explain")


async def test_estimate_level_parses_json() -> None:
    llm = ScriptedLLM([ScriptedResponse(['{"level": "提高"}'])])
    level = await estimate_level(cast(LLMClient, llm), question="q", reply="r")
    assert level == "提高"
    assert llm.calls[0]["temperature"] == 0.0


async def test_estimate_level_falls_back_to_default() -> None:
    llm = ScriptedLLM([ScriptedResponse(["任意文本"])])
    level = await estimate_level(cast(LLMClient, llm), question="q", reply="r")
    assert level == DEFAULT_LEVEL


# ── rag 工具接线（讲解阶段溯源）────────────────────────────────────────

class StubRagTool:
    """返回预置 ToolResult 的假 rag 工具，并记录 execute 调用参数。"""

    name = "rag"

    def __init__(self, result: ToolResult | None = None, error: Exception | None = None) -> None:
        self._result = result if result is not None else ToolResult(success=False)
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


RAG_HITS = ToolResult(
    content="找到 2 条与「罗尔定理证明题」相关的知识点：\n[1] …",
    sources=[
        {
            "title": "05-微分中值定理·罗尔定理",
            "excerpt": "罗尔定理：三个条件缺一不可…",
            "score": 0.92,
            "type": "rag_knowledge",
        },
        {
            "title": "05-微分中值定理·辅助函数构造",
            "excerpt": "构造辅助函数是使用罗尔定理的核心技巧…",
            "score": 0.87,
            "type": "rag_knowledge",
        },
    ],
    metadata={"query": "罗尔定理证明题", "top_k": 5, "scores": [0.92, 0.87]},
    success=True,
)


def _registry_with_rag(rag_tool: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(rag_tool)
    return registry


async def test_explain_uses_rag_references_and_emits_sources() -> None:
    rag_tool = StubRagTool(RAG_HITS)
    cap, llm, _ = _capability(
        HAPPY_PATH, tools=_registry_with_rag(cast(BaseTool, rag_tool))
    )
    ctx = TurnContext(user_message="罗尔定理证明题")
    events, error = await _drive(cap, ctx, ["卡在第一步", "是 0/0 型吧", "用洛必达"])

    assert error is None
    # SOURCES 事件：讲解 STAGE_START 之后、讲解第一条 content 之前
    source_events = [e for e in events if e.type == SOURCES]
    assert len(source_events) == 1
    ev = source_events[0]
    assert ev.stage == "讲解"
    assert ev.metadata["sources"] == RAG_HITS.sources
    assert ev.metadata["query"] == "罗尔定理证明题"
    explain_start = next(
        i for i, e in enumerate(events) if e.type == STAGE_START and e.stage == "讲解"
    )
    assert events.index(ev) == explain_start + 1
    # 讲解 user 消息拼接了参考资料模板与命中摘要
    explain_call = llm.calls[6]
    assert REFERENCES_PROMPT.split("\n")[0] in explain_call["messages"][1]["content"]
    assert "罗尔定理" in explain_call["messages"][1]["content"]
    # 检索不新增 LLM 调用（仍为诊断/估档位/提示×2/判答×2/讲解/巩固 8 次）
    assert len(llm.calls) == 8
    # 检索查询词 = 题目原文
    assert rag_tool.calls == [{"query": "罗尔定理证明题"}]


async def test_explain_without_rag_tool_silently_skips() -> None:
    cap, llm, _ = _capability(HAPPY_PATH, tools=ToolRegistry())
    events, error = await _drive(
        cap, TurnContext(user_message="求极限题"), ["卡住", "对", "对"]
    )
    assert error is None
    assert not any(e.type == SOURCES for e in events)
    assert "参考资料" not in llm.calls[6]["messages"][1]["content"]


async def test_explain_rag_failure_silently_skips() -> None:
    rag_tool = StubRagTool(ToolResult(content="知识库索引尚未构建", success=False))
    cap, llm, _ = _capability(
        HAPPY_PATH, tools=_registry_with_rag(cast(BaseTool, rag_tool))
    )
    events, error = await _drive(
        cap, TurnContext(user_message="求极限题"), ["卡住", "对", "对"]
    )
    assert error is None
    assert not any(e.type == SOURCES for e in events)
    assert "参考资料" not in llm.calls[6]["messages"][1]["content"]
    assert rag_tool.calls == [{"query": "求极限题"}]


async def test_explain_rag_exception_silently_skips() -> None:
    rag_tool = StubRagTool(error=RuntimeError("工具内部崩溃"))
    cap, _, _ = _capability(
        HAPPY_PATH, tools=_registry_with_rag(cast(BaseTool, rag_tool))
    )
    events, error = await _drive(
        cap, TurnContext(user_message="求极限题"), ["卡住", "对", "对"]
    )
    assert error is None  # 工具异常不打断教学主流程
    assert not any(e.type == SOURCES for e in events)


async def test_collect_references_no_tool() -> None:
    assert await _collect_references(None, "题") == ("", [])
    assert await _collect_references(ToolRegistry(), "题") == ("", [])


# ── question_bank / check_answer 接线（W6）─────────────────────────────

class StubBankTool:
    """返回预置 ToolResult 的假 question_bank 工具，并记录 execute 调用。"""

    name = "question_bank"

    def __init__(self, result: ToolResult | None = None, error: Exception | None = None) -> None:
        self._result = result if result is not None else ToolResult(success=False)
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


class StubCheckTool:
    """返回预置 ToolResult 的假 check_answer 工具（不产生 LLM 调用）。"""

    name = "check_answer"

    def __init__(self, result: ToolResult | None = None) -> None:
        self._result = result if result is not None else ToolResult(success=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        return self._result


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


BANK_ITEM = {
    "id": "qb-05-常规-01",
    "topic": "微分中值定理",
    "level": "常规",
    "question": (
        "设 $f(x)$ 在 $[0,1]$ 上连续、在 $(0,1)$ 内可导，且 $f(0)=0$，$f(1)=1$。"
        "证明：存在 $\\xi\\in(0,1)$，使 $f'(\\xi)=2\\xi$。"
    ),
    "answer": "构造 $F(x)=f(x)-x^2$，由罗尔定理得证。",
}


def _bank_hit_result() -> ToolResult:
    return ToolResult(
        content=BANK_ITEM["question"],
        metadata={"item": BANK_ITEM, "item_id": BANK_ITEM["id"], "position": 0, "total": 1},
        success=True,
    )


async def test_consolidate_without_question_bank_no_extraction_call() -> None:
    """硬守卫：无 question_bank 工具时零新增 LLM 调用（calls==8）。"""
    cap, llm, _ = _capability(HAPPY_PATH, tools=ToolRegistry())
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])
    assert error is None
    assert len(llm.calls) == 8
    assert not any(e.type == SOURCES for e in events)


async def test_consolidate_bank_hit_streams_bank_question_and_summary_only() -> None:
    bank_tool = StubBankTool(_bank_hit_result())
    responses = list(HAPPY_PATH)
    # 巩固的 LLM 出题调用被替换：第 7 响应改提取、追加总结响应
    responses[7] = ScriptedResponse(['{"topic": "微分中值定理"}'])
    responses.append(ScriptedResponse(["一句话总结：辅助函数与罗尔定理。"]))
    cap, llm, _ = _capability(responses, tools=_registry_with(cast(BaseTool, bank_tool)))
    events, error = await _drive(cap, TurnContext(user_message="罗尔型证明题"), ["卡", "对", "对"])

    assert error is None
    # 调用数：诊断/估档位/提示×2/判答×2/讲解/提取/总结 = 9
    assert len(llm.calls) == 9
    # 提取调用低温且查询词=题目原文
    assert llm.calls[7]["temperature"] == 0.0
    assert "罗尔型证明题" in llm.calls[7]["messages"][0]["content"]
    # 题库收到 (topic, level)，level 来自诊断估档位（HAPPY_PATH=常规）
    assert bank_tool.calls == [{"topic": "微分中值定理", "level": "常规"}]
    # 巩固 stage：题库题直发总线 + 总结调用不含出题指令
    consolidate = [e for e in events if e.stage == "巩固"]
    assert any(e.type == CONTENT and e.content.startswith("变式题：") for e in consolidate)
    assert "不要重复出新题" in llm.calls[8]["messages"][1]["content"]
    assert "出一道与本题同考点" not in llm.calls[8]["messages"][1]["content"]


async def test_consolidate_bank_miss_falls_back_to_llm_variant() -> None:
    bank_tool = StubBankTool(
        ToolResult(content="暂未收录", metadata={"item": None}, success=True)
    )
    responses = list(HAPPY_PATH)
    responses[7] = ScriptedResponse(['{"topic": "泰勒公式"}'])
    responses.append(ScriptedResponse(["变式题与总结（LLM 生成）。"]))
    cap, llm, _ = _capability(responses, tools=_registry_with(cast(BaseTool, bank_tool)))
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])

    assert error is None
    # 提取发生 + 查库 miss → 回退原 CONSOLIDATE_PROMPT：9 次调用
    assert len(llm.calls) == 9
    assert "出一道与本题同考点" in llm.calls[8]["messages"][1]["content"]
    # 未出现题库直发内容
    assert not any(e.content.startswith("变式题：") for e in events if e.type == CONTENT)


async def test_consolidate_topic_extraction_failure_falls_back() -> None:
    bank_tool = StubBankTool(_bank_hit_result())
    responses = list(HAPPY_PATH)
    responses[7] = ScriptedResponse(["这题考点很多，不好说"])  # 非 JSON → 提取失败
    responses.append(ScriptedResponse(["变式题与总结（LLM 生成）。"]))
    cap, llm, _ = _capability(responses, tools=_registry_with(cast(BaseTool, bank_tool)))
    _, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])

    assert error is None
    assert len(llm.calls) == 9
    assert bank_tool.calls == []  # 提取失败后不查库
    assert "出一道与本题同考点" in llm.calls[8]["messages"][1]["content"]


async def test_consolidate_bank_tool_exception_falls_back() -> None:
    bank_tool = StubBankTool(error=RuntimeError("题库崩溃"))
    responses = list(HAPPY_PATH)
    responses[7] = ScriptedResponse(['{"topic": "泰勒公式"}'])
    responses.append(ScriptedResponse(["变式题与总结（LLM 生成）。"]))
    cap, llm, _ = _capability(responses, tools=_registry_with(cast(BaseTool, bank_tool)))
    _, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])

    assert error is None  # 工具异常静默回退，不打断教学
    assert len(llm.calls) == 9


async def test_consolidate_bank_tool_absent_but_registry_has_others() -> None:
    """只注册 check_answer：question_bank 缺失 → 不发起提取（calls==8）。"""
    responses = list(HAPPY_PATH)
    cap, llm, _ = _capability(
        responses, tools=_registry_with(cast(BaseTool, StubCheckTool()))
    )
    _, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])
    assert error is None
    assert len(llm.calls) == 8


async def test_judge_with_check_answer_tool_same_call_count() -> None:
    """真 CheckAnswerTool 端到端：判答经工具，调用数与温度断言全不变。"""
    from nnnu.tools.check_answer import CheckAnswerTool

    llm = ScriptedLLM(list(HAPPY_PATH))
    cap = MathTutorCapability(
        llm=cast(LLMClient, llm),
        usage=cast(UsageTracker, StubTracker()),
        tools=_registry_with(CheckAnswerTool(llm=cast(LLMClient, llm))),
    )
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "对", "对"])

    assert error is None
    assert len(llm.calls) == 8
    temps = [call["temperature"] for call in llm.calls]
    assert temps[1] == 0.0 and temps[3] == 0.0 and temps[5] == 0.0
    assert [e.stage for e in events if e.type == STAGE_START] == [
        "诊断",
        "引导",
        "讲解",
        "巩固",
    ]


async def test_judge_tool_unparseable_metadata_falls_back_direct() -> None:
    stub = StubCheckTool(ToolResult(content="{}", metadata={}, success=True))
    cap, llm, _ = _capability(GIVE_UP_PATH, tools=_registry_with(cast(BaseTool, stub)))
    events, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "直接讲吧"])

    assert error is None
    # 工具结果不可解析 → 回退直调判答：GIVE_UP_PATH 6 次调用不变
    assert len(llm.calls) == 6
    assert [e.stage for e in events if e.type == STAGE_START] == [
        "诊断",
        "引导",
        "讲解",
        "巩固",
    ]


async def test_judge_tool_failure_falls_back_direct() -> None:
    stub = StubCheckTool(ToolResult(content="失败", success=False))
    cap, llm, _ = _capability(GIVE_UP_PATH, tools=_registry_with(cast(BaseTool, stub)))
    _, error = await _drive(cap, TurnContext(user_message="题"), ["卡", "直接讲吧"])

    assert error is None
    assert len(llm.calls) == 6


def test_topic_prompt_lists_all_topics() -> None:
    from nnnu.capabilities.math.prompts import TOPIC_JSON_PROMPT
    from nnnu.capabilities.math.topics import TOPICS

    prompt = TOPIC_JSON_PROMPT.format(question="任意题")
    for topic in TOPICS:
        assert topic in prompt
