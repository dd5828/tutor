"""core/events.py 单测。"""

from __future__ import annotations

import time

from nnnu.core.events import StreamEvent, StreamEventType


def test_event_type_values() -> None:
    """v1 保留的 10 种事件枚举值（路线图 §4.1）。"""
    assert {t.value for t in StreamEventType} == {
        "stage_start",
        "stage_end",
        "content",
        "tool_call",
        "tool_result",
        "sources",
        "result",
        "error",
        "done",
        "wait_for_input",
    }


def test_to_dict_serializes_enum_as_string() -> None:
    event = StreamEvent(type=StreamEventType.CONTENT, source="chat", content="你好")
    data = event.to_dict()
    assert data["type"] == "content"
    assert isinstance(data["type"], str)  # 字符串枚举值，而非枚举对象
    assert data["source"] == "chat"
    assert data["content"] == "你好"


def test_defaults_and_metadata_isolation() -> None:
    e1 = StreamEvent(type=StreamEventType.DONE)
    e2 = StreamEvent(type=StreamEventType.DONE)
    assert e1.metadata == {} and e2.metadata == {}
    e1.metadata["k"] = "v"
    assert e2.metadata == {}  # default_factory 保证实例间不共享
    assert isinstance(e1.timestamp, float)
    assert e1.seq == 0
    assert e1.session_id == "" and e1.turn_id == ""


def test_timestamp_default_is_now() -> None:
    before = time.time()
    event = StreamEvent(type=StreamEventType.CONTENT)
    after = time.time()
    assert before <= event.timestamp <= after
