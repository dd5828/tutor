"""WebSocket /chat 回环测试（TestClient，走 echo 假能力，不触 LLM）。"""

from __future__ import annotations

import json
from typing import Any, cast

from fastapi.testclient import TestClient

from nnnu.api.main import create_app


def _recv(ws: Any) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(ws.receive_text()))


def test_chat_ws_echo_roundtrip() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"message": "你好", "capability": "echo"})
        events = [_recv(ws) for _ in range(4)]

    assert [e["type"] for e in events] == ["stage_start", "content", "stage_end", "done"]
    assert events[1]["content"] == "echo: 你好"
    assert events[1]["source"] == "echo"
    assert events[3]["metadata"] == {"status": "completed"}


def test_chat_ws_unknown_capability_emits_error_flow() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"message": "hi", "capability": "ghost"})
        events = [_recv(ws) for _ in range(2)]

    assert [e["type"] for e in events] == ["error", "done"]
    assert "未知能力: ghost" in events[0]["content"]
    assert events[1]["metadata"] == {"status": "failed"}


def test_chat_ws_invalid_json_emits_error_event() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/chat") as ws:
        ws.send_text("这不是 JSON")
        event = _recv(ws)

    assert event["type"] == "error"
    assert event["source"] == "api"
    assert event["content"] == "无效 JSON"
