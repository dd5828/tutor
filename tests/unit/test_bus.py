"""core/bus.py 单测：订阅/广播/关闭/重放 + 反问等待 + 阶段 + NDJSON。"""

from __future__ import annotations

import asyncio
import json

import pytest

from nnnu.core.bus import StreamBus
from nnnu.core.events import StreamEvent, StreamEventType


async def collect(bus: StreamBus) -> list[StreamEvent]:
    """订阅并收齐事件（要求调用方最终 close）。"""
    return [event async for event in bus.subscribe()]


async def test_broadcast_to_multiple_subscribers() -> None:
    bus = StreamBus()
    task1 = asyncio.create_task(collect(bus))
    task2 = asyncio.create_task(collect(bus))
    await asyncio.sleep(0)  # 让两个订阅者完成注册
    await bus.content("你好")
    await bus.close()
    got1, got2 = await asyncio.gather(task1, task2)
    assert [e.content for e in got1] == ["你好"]
    assert got1 == got2  # 扇出：每个订阅者收到同一事件对象


async def test_replay_delivers_past_events() -> None:
    bus = StreamBus()
    await bus.content("a")
    await bus.content("b")
    await bus.close()
    events = await collect(bus)  # 订阅发生在 emit 之后
    assert [e.content for e in events] == ["a", "b"]


async def test_replay_then_live_has_no_duplicates() -> None:
    bus = StreamBus()
    await bus.content("old")
    task = asyncio.create_task(collect(bus))
    await asyncio.sleep(0)  # 订阅完成、重放结束后再发新事件
    await bus.content("new")
    await bus.close()
    events = await task
    assert [e.content for e in events] == ["old", "new"]


async def test_close_ends_iteration() -> None:
    bus = StreamBus()
    await bus.content("x")
    await bus.close()
    events = await collect(bus)
    assert [e.content for e in events] == ["x"]


async def test_subscribe_after_close_replays_then_ends() -> None:
    bus = StreamBus()
    await bus.content("x")
    await bus.close()
    events = await collect(bus)  # 关闭后才订阅：重放历史后立即结束
    assert [e.content for e in events] == ["x"]


async def test_emit_after_close_is_dropped() -> None:
    bus = StreamBus()
    await bus.close()
    await bus.content("ghost")
    events = await collect(bus)
    assert events == []


async def test_mark_closed_is_sync_and_ends_subscribers() -> None:
    bus = StreamBus()
    task = asyncio.create_task(collect(bus))
    await asyncio.sleep(0)
    bus.mark_closed()  # 同步调用，无需 await
    events = await task
    assert events == []


async def test_stage_emits_start_and_end() -> None:
    bus = StreamBus()
    async with bus.stage("引导", source="tutor"):
        await bus.content("第一步")
    await bus.close()
    events = await collect(bus)
    assert [e.type for e in events] == [
        StreamEventType.STAGE_START,
        StreamEventType.CONTENT,
        StreamEventType.STAGE_END,
    ]
    assert events[0].stage == "引导" == events[2].stage
    assert events[0].source == "tutor" == events[2].source


async def test_stage_end_emitted_on_exception() -> None:
    bus = StreamBus()
    with pytest.raises(RuntimeError, match="boom"):
        async with bus.stage("引导"):
            raise RuntimeError("boom")
    await bus.close()
    events = await collect(bus)
    assert [e.type for e in events] == [StreamEventType.STAGE_START, StreamEventType.STAGE_END]


async def test_tool_call_and_result_metadata() -> None:
    bus = StreamBus()
    await bus.tool_call("check_answer", {"answer": "2"})
    await bus.tool_result("check_answer", "错误，第三步混淆", metadata={"ok": True})
    await bus.close()
    events = await collect(bus)
    call, res = events
    assert call.type == StreamEventType.TOOL_CALL
    assert call.content == "check_answer"
    assert call.metadata["args"] == {"answer": "2"}
    assert res.metadata["tool"] == "check_answer"
    assert res.metadata["ok"] is True


async def test_sources_and_result_metadata() -> None:
    bus = StreamBus()
    await bus.sources([{"title": "课本第3章"}], source="rag")
    await bus.result({"score": 8}, metadata={"done": True})
    await bus.close()
    events = await collect(bus)
    assert events[0].type == StreamEventType.SOURCES
    assert events[0].metadata["sources"] == [{"title": "课本第3章"}]
    assert events[1].type == StreamEventType.RESULT
    assert events[1].metadata["score"] == 8
    assert events[1].metadata["done"] is True


async def test_error_event() -> None:
    bus = StreamBus()
    await bus.error("未知能力: x", source="orchestrator")
    await bus.close()
    events = await collect(bus)
    assert events[0].type == StreamEventType.ERROR
    assert events[0].content == "未知能力: x"


async def test_metadata_extra_overrides_base() -> None:
    """调用方 metadata 覆盖协议字段（extra 优先）。"""
    bus = StreamBus()
    await bus.tool_call("t", {"a": 1}, metadata={"args": "override"})
    await bus.close()
    events = await collect(bus)
    assert events[0].metadata["args"] == "override"


async def test_wait_for_input_returns_submitted_input() -> None:
    bus = StreamBus()
    got = asyncio.create_task(bus.wait_for_input("你的答案是什么？", source="tutor"))
    await asyncio.sleep(0)  # 等 wait_for_input 挂起
    bus.submit_input("x=2")
    assert await got == "x=2"
    await bus.close()
    events = await collect(bus)
    assert [e.type for e in events] == [StreamEventType.WAIT_FOR_INPUT]
    assert events[0].content == "你的答案是什么？"


async def test_wait_for_input_timeout_returns_empty() -> None:
    bus = StreamBus()
    result = await bus.wait_for_input("在吗？", timeout=0)
    assert result == ""


async def test_max_history_bounds_replay() -> None:
    bus = StreamBus(max_history=2)
    for i in range(5):
        await bus.content(str(i))
    await bus.close()
    events = await collect(bus)
    assert [e.content for e in events] == ["3", "4"]


def test_event_to_json_is_single_line_utf8() -> None:
    event = StreamEvent(type=StreamEventType.CONTENT, content="你好，世界")
    line = StreamBus.event_to_json(event)
    assert "\n" not in line
    assert "你好，世界" in line  # ensure_ascii=False：中文不转义
    data = json.loads(line)
    assert data["type"] == "content"
    assert data["content"] == "你好，世界"
