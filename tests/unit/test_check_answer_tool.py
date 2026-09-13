"""CheckAnswerTool 单测（复用 ScriptedLLM 预编排，不触网）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from nnnu.capabilities.math.pedagogy import Judgement
from nnnu.core import ChatError
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo
from nnnu.tools.check_answer import CheckAnswerTool
from tests.unit.test_math_capability import ScriptedLLM, ScriptedResponse


def test_get_definition_schema() -> None:
    definition = CheckAnswerTool().get_definition()
    assert definition.name == "check_answer"
    assert {p.name for p in definition.parameters} == {"question", "last_hint", "reply"}
    assert all(p.required for p in definition.parameters)
    # on_usage 是内部记账参数，不进 schema
    schema = definition.to_openai_schema()
    assert "on_usage" not in schema["function"]["parameters"]["properties"]


async def test_execute_returns_json_content_and_metadata() -> None:
    llm = ScriptedLLM(
        [ScriptedResponse(['{"correct": false, "error_layer": "公式", "next_step": "continue"}'])]
    )
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    result = await tool.execute(question="题", last_hint="提示", reply="学生答")
    assert result.success
    assert json.loads(result.content) == {
        "correct": False,
        "error_layer": "公式",
        "next_step": "continue",
    }
    assert result.metadata["judgement"] == {
        "correct": False,
        "error_layer": "公式",
        "next_step": "continue",
    }
    # 判答走低温结构化调用，题目与作答都进了提示词
    assert llm.calls[0]["temperature"] == 0.0
    assert "题" in llm.calls[0]["messages"][0]["content"]
    assert "学生答" in llm.calls[0]["messages"][0]["content"]


async def test_execute_empty_input_fails_without_llm_call() -> None:
    llm = ScriptedLLM([])  # 零响应：任何 LLM 调用都会断言失败
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    result = await tool.execute(question="  ", last_hint="h", reply="r")
    assert not result.success
    result2 = await tool.execute(question="q", last_hint="h", reply=" ")
    assert not result2.success
    assert llm.calls == []


async def test_execute_malformed_json_conservative() -> None:
    llm = ScriptedLLM([ScriptedResponse(["抱歉我没法判断"])])
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    result = await tool.execute(question="q", last_hint="h", reply="r")
    # 工具路径同样保守：未解出、进入讲解
    assert result.metadata["judgement"] == {
        "correct": False,
        "error_layer": "",
        "next_step": "explain",
    }


async def test_execute_llm_error_propagates() -> None:
    llm = ScriptedLLM([ScriptedResponse([], error=ChatError("模型超时"))])
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    # 传播语义：判答故障不吞错（与 rag_tool 相反）
    with pytest.raises(ChatError):
        await tool.execute(question="q", last_hint="h", reply="r")


class _RecorderLLM:
    """记录 kwargs 并触发 on_usage 回调的最小假 LLM。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls.append({"messages": messages, **kwargs})
        on_usage = kwargs.get("on_usage")
        if callable(on_usage):
            on_usage(UsageInfo(1, 2, 3))
        yield '{"correct": true, "error_layer": "无", "next_step": "solved"}'


async def test_execute_passes_on_usage() -> None:
    llm = _RecorderLLM()
    received: list[UsageInfo] = []
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    result = await tool.execute(question="q", last_hint="h", reply="r", on_usage=received.append)
    assert result.success
    # 回调经 工具→judge_answer→chat_stream 链路透传
    assert received == [UsageInfo(1, 2, 3)]


async def test_judgement_roundtrip_matches_dataclass() -> None:
    llm = ScriptedLLM(
        [ScriptedResponse(['{"correct": true, "error_layer": "无", "next_step": "solved"}'])]
    )
    tool = CheckAnswerTool(llm=cast(LLMClient, llm))
    result = await tool.execute(question="q", last_hint="h", reply="r")
    assert Judgement(**result.metadata["judgement"]) == Judgement(
        correct=True, error_layer="", next_step="solved"
    )
