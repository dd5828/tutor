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


def test_chat_ws_quiz_roundtrip_with_reply() -> None:
    """反问教学循环端到端：出题 → wait_for_input → 发答案 → 批改 → done。"""
    client = TestClient(create_app())
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"message": "开始", "capability": "quiz"})
        before = []
        while True:
            event = _recv(ws)
            before.append(event)
            if event["type"] == "wait_for_input":
                break

        assert [e["type"] for e in before] == [
            "stage_start",
            "content",
            "stage_end",
            "stage_start",
            "wait_for_input",
        ]
        assert before[1]["content"].startswith("第 1 题")
        assert before[4]["content"] == "请输入你的答案"

        ws.send_text("56")  # 纯文本回复
        after = []
        while True:
            event = _recv(ws)
            after.append(event)
            if event["type"] == "done":
                break

        assert [e["type"] for e in after] == ["content", "stage_end", "done"]
        assert after[0]["content"] == "正确！"
        assert after[2]["metadata"] == {"status": "completed"}


def test_index_html_served_at_root() -> None:
    client = TestClient(create_app())

    resp = client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "nnnu 辅导 Demo" in resp.text
