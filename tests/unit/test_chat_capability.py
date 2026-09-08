"""ChatCapability 单测（注入假 LLMClient，不触网）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from nnnu.capabilities.chat.capability import ChatCapability
from nnnu.capabilities.chat.prompts import SYSTEM_PROMPT
from nnnu.core import ChatError, StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.services.llm import LLMClient


class StubLLM:
    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.calls: list[list[dict[str, str]]] = []

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


def _capability(
    chunks: list[str], error: Exception | None = None
) -> tuple[ChatCapability, StubLLM]:
    stub = StubLLM(chunks, error)
    return ChatCapability(llm=cast(LLMClient, stub)), stub


async def _collect_run(cap: ChatCapability, ctx: TurnContext) -> list[StreamEvent]:
    bus = StreamBus()
    await cap.run(ctx, bus)
    bus.mark_closed()  # 生产由 orchestrator 关闭；这里手动关以便重放
    return [event async for event in bus.subscribe()]


async def test_run_emits_stage_and_content_events() -> None:
    cap, _ = _capability(["你", "好"])

    events = await _collect_run(cap, TurnContext(user_message="hi"))

    assert [e.type for e in events] == [
        StreamEventType.STAGE_START,
        StreamEventType.CONTENT,
        StreamEventType.CONTENT,
        StreamEventType.STAGE_END,
    ]
    assert "".join(e.content for e in events) == "你好"
    assert all(e.source == "chat" for e in events)


async def test_run_builds_messages_with_system_prompt_and_history() -> None:
    cap, stub = _capability(["x"])
    ctx = TurnContext(
        user_message="这题怎么做",
        conversation_history=[{"role": "assistant", "content": "上次讲到二次函数"}],
    )

    await _collect_run(cap, ctx)

    messages = stub.calls[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "assistant", "content": "上次讲到二次函数"}
    assert messages[-1] == {"role": "user", "content": "这题怎么做"}


async def test_run_llm_error_propagates_but_stage_pair_closes() -> None:
    cap, _ = _capability(["半句"], error=ChatError("模型超时"))
    bus = StreamBus()

    with pytest.raises(ChatError):
        await cap.run(TurnContext(user_message="hi"), bus)

    bus.mark_closed()
    events = [event async for event in bus.subscribe()]
    # 异常也不破坏 stage 配对：已流出的部分 + STAGE_END 都在
    assert [e.type for e in events] == [
        StreamEventType.STAGE_START,
        StreamEventType.CONTENT,
        StreamEventType.STAGE_END,
    ]
    assert events[1].content == "半句"
