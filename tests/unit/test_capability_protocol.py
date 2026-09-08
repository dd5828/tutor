"""core/capability_protocol.py 单测。"""

from __future__ import annotations

import pytest

from nnnu.core.bus import StreamBus
from nnnu.core.capability_protocol import BaseCapability, CapabilityManifest
from nnnu.core.context import TurnContext
from nnnu.core.events import StreamEvent, StreamEventType


async def collect(bus: StreamBus) -> list[StreamEvent]:
    """订阅并收齐事件（要求调用方最终 close）。"""
    return [event async for event in bus.subscribe()]


class EchoCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="echo",
        description="原样回显用户消息",
        stages=["回显"],
        tools_used=[],
    )

    async def run(self, context: TurnContext, bus: StreamBus) -> None:
        await bus.content(context.user_message, source=self.name)


def test_manifest_defaults_and_isolation() -> None:
    m1 = CapabilityManifest(name="a", description="b")
    m2 = CapabilityManifest(name="c", description="d")
    assert m1.stages == [] and m1.tools_used == []
    m1.stages.append("阶段1")
    assert m2.stages == []


async def test_run_emits_content_on_bus() -> None:
    """能力通过总线汇报进度的最小闭环（协议约定的调用方式）。"""
    bus = StreamBus()
    await EchoCapability().run(TurnContext(user_message="你好"), bus)
    await bus.close()
    events = await collect(bus)
    assert [e.type for e in events] == [StreamEventType.CONTENT]
    assert events[0].content == "你好"
    assert events[0].source == "echo"


def test_name_and_stages_properties() -> None:
    cap = EchoCapability()
    assert cap.name == "echo"
    assert cap.stages == ["回显"]


def test_base_capability_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseCapability()  # type: ignore[abstract]
