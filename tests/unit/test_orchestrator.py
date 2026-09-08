"""Orchestrator.handle() 单测：路由 / 并发流式 / 异常反射。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import cast

from nnnu.core import (
    BaseCapability,
    CapabilityManifest,
    ChatError,
    StreamBus,
    StreamEvent,
    StreamEventType,
    TurnContext,
)
from nnnu.runtime.orchestrator import Orchestrator
from nnnu.runtime.registry import CapabilityRegistry


class StubCapability(BaseCapability):
    """按名字注册的假能力；fail 非空时 run 抛异常。"""

    def __init__(self, name: str, fail: Exception | None = None) -> None:
        self.manifest = CapabilityManifest(name=name, description="测试能力")
        self._fail = fail
        self.runs: list[tuple[TurnContext, StreamBus]] = []

    async def run(self, context: TurnContext, bus: StreamBus) -> None:
        self.runs.append((context, bus))
        if self._fail is not None:
            raise self._fail
        await bus.content(f"{self.name}: {context.user_message}", source=self.name)


class SlowCapability(BaseCapability):
    """先发一个事件，再停顿，最后发第二个事件。"""

    def __init__(self) -> None:
        self.manifest = CapabilityManifest(name="slow", description="测试能力")
        self.finished = asyncio.Event()

    async def run(self, context: TurnContext, bus: StreamBus) -> None:
        await bus.content("a", source="slow")
        await asyncio.sleep(0.05)
        self.finished.set()
        await bus.content("b", source="slow")


async def _collect(agen: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in agen]


def _registry_with(*caps: BaseCapability) -> tuple[Orchestrator, CapabilityRegistry]:
    registry = CapabilityRegistry()
    for cap in caps:
        registry.register(cap)
    return Orchestrator(caps=registry), registry


async def test_handle_routes_to_active_capability() -> None:
    orch, registry = _registry_with(StubCapability("echo"))
    ctx = TurnContext(session_id="s1", user_message="你好", active_capability="echo")

    events = await _collect(orch.handle(ctx))

    assert [e.type for e in events] == [StreamEventType.CONTENT, StreamEventType.DONE]
    assert events[0].content == "echo: 你好"
    assert events[0].source == "echo"
    assert events[1].metadata == {"status": "completed"}
    cap = cast(StubCapability, registry.get("echo"))
    assert cap.runs[0][0] is ctx


async def test_handle_defaults_to_chat_when_active_capability_none() -> None:
    orch, _ = _registry_with(StubCapability("chat"))
    ctx = TurnContext(user_message="hi")  # active_capability 未指定

    events = await _collect(orch.handle(ctx))

    assert events[0].content == "chat: hi"
    assert events[-1].metadata == {"status": "completed"}


async def test_handle_fills_session_id_when_empty() -> None:
    orch, _ = _registry_with(StubCapability("chat"))
    ctx = TurnContext(user_message="hi")

    await _collect(orch.handle(ctx))

    assert ctx.session_id
    uuid.UUID(ctx.session_id)  # 合法 UUID


async def test_handle_keeps_existing_session_id() -> None:
    orch, _ = _registry_with(StubCapability("chat"))
    ctx = TurnContext(session_id="keep-me", user_message="hi")

    await _collect(orch.handle(ctx))

    assert ctx.session_id == "keep-me"


async def test_handle_unknown_capability_emits_error_then_done_failed() -> None:
    orch, _ = _registry_with(StubCapability("chat"))
    ctx = TurnContext(active_capability="ghost")

    events = await _collect(orch.handle(ctx))

    assert [e.type for e in events] == [StreamEventType.ERROR, StreamEventType.DONE]
    assert events[0].source == "orchestrator"
    assert "未知能力: ghost" in events[0].content
    assert "chat" in events[0].content  # 附带可用能力清单
    assert events[0].metadata == {"turn_terminal": True, "status": "failed"}
    assert events[1].metadata == {"status": "failed"}


async def test_handle_capability_exception_emits_error_and_done_failed() -> None:
    orch, _ = _registry_with(StubCapability("boom", fail=ValueError("炸了")))
    ctx = TurnContext(active_capability="boom")

    events = await _collect(orch.handle(ctx))

    assert [e.type for e in events] == [StreamEventType.ERROR, StreamEventType.DONE]
    assert events[0].source == "boom"
    assert events[0].content == "炸了"
    # 普通异常没有反射属性，metadata 只含协议字段
    assert events[0].metadata == {"turn_terminal": True, "status": "failed"}
    assert events[1].metadata == {"status": "failed"}


async def test_handle_reflects_chat_error_attributes() -> None:
    fail = ChatError(
        "模型超时",
        retryable=True,
        partial_response="已经输出了半句",
        error_code="LLM_TIMEOUT",
    )
    orch, _ = _registry_with(StubCapability("boom", fail=fail))
    ctx = TurnContext(active_capability="boom")

    events = await _collect(orch.handle(ctx))

    metadata = events[0].metadata
    assert metadata["retryable"] is True
    assert metadata["partial_response"] == "已经输出了半句"
    assert metadata["error_code"] == "LLM_TIMEOUT"


async def test_handle_yields_first_event_before_capability_finishes() -> None:
    """并发结构验证：能力还在跑时消费者已收到首个事件。"""
    slow = SlowCapability()
    orch, _ = _registry_with(slow)

    agen = orch.handle(TurnContext(active_capability="slow"))
    first = await anext(agen)
    assert not slow.finished.is_set()  # 能力尚未跑完，事件已流出
    assert first.type is StreamEventType.CONTENT
    assert first.content == "a"

    rest = [event async for event in agen]
    assert [e.type for e in rest] == [StreamEventType.CONTENT, StreamEventType.DONE]
    assert rest[0].content == "b"
    assert rest[1].metadata == {"status": "completed"}
    assert slow.finished.is_set()


async def test_handle_uses_injected_registry_not_global() -> None:
    # 注入独立注册表：即使全局注册表为空，路由照常工作
    orch, _ = _registry_with(StubCapability("chat"))

    events = await _collect(orch.handle(TurnContext(user_message="hi")))

    assert events[-1].metadata == {"status": "completed"}
