"""WebSocket /chat：收消息 → orchestrator → 逐事件 NDJSON 推回。

客户端协议：发单行 JSON（message 必填，session_id / capability 可选），
收 NDJSON 事件流（见 core/events.py），done 事件标志本轮结束。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from nnnu.core import StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.runtime.orchestrator import Orchestrator

router = APIRouter()


@router.websocket("/chat")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator = Orchestrator()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    StreamBus.event_to_json(
                        StreamEvent(
                            type=StreamEventType.ERROR,
                            source="api",
                            content="无效 JSON",
                            metadata={"turn_terminal": True},
                        )
                    )
                )
                continue
            ctx = TurnContext(
                user_message=str(payload.get("message", "")),
                session_id=str(payload.get("session_id", "")),
                active_capability=payload.get("capability"),
            )
            async for event in orchestrator.handle(ctx):
                await websocket.send_text(StreamBus.event_to_json(event))
    except WebSocketDisconnect:
        pass
