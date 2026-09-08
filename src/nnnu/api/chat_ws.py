"""WebSocket /chat：收消息 → orchestrator → 逐事件 NDJSON 推回。

客户端协议：
- 开启新轮次：发单行 JSON（message 必填，session_id / capability 可选）
- 轮次进行中收到 wait_for_input 事件后，直接发**纯文本**作为回复
- 轮次进行中发送的任何消息都会被当作反问回复投递
- 服务端每个事件回一行 NDJSON，done 事件标志本轮结束
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from nnnu.core import StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.runtime.orchestrator import Orchestrator

router = APIRouter()


def _build_context(raw: str) -> TurnContext | None:
    """解析新轮次 JSON；无效 JSON 返回 None。"""
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return TurnContext(
        user_message=str(payload.get("message", "")),
        session_id=str(payload.get("session_id", "")),
        active_capability=payload.get("capability"),
    )


async def _forward_turn(
    websocket: WebSocket,
    orchestrator: Orchestrator,
    ctx: TurnContext,
    reply_queue: asyncio.Queue[str],
) -> None:
    """跑一轮：orchestrator 事件逐条 NDJSON 推回客户端。"""
    async for event in orchestrator.handle(ctx, reply_queue=reply_queue):
        await websocket.send_text(StreamBus.event_to_json(event))


@router.websocket("/chat")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    orchestrator = Orchestrator()
    try:
        while True:
            # 等待开启新轮次的消息
            raw = await websocket.receive_text()
            ctx = _build_context(raw)
            if ctx is None:
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

            reply_queue: asyncio.Queue[str] = asyncio.Queue()
            forward_task = asyncio.create_task(
                _forward_turn(websocket, orchestrator, ctx, reply_queue)
            )
            try:
                # 轮次进行中：后续消息全部作为反问回复投递；轮次结束即停止接收，
                # 未读的消息留在 WS 缓冲，下一轮作为新轮次首条消息（无丢失竞态）
                while True:
                    recv_task = asyncio.create_task(websocket.receive_text())
                    done, _ = await asyncio.wait(
                        {recv_task, forward_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if recv_task in done:
                        reply_queue.put_nowait(recv_task.result())
                    if forward_task in done:
                        recv_task.cancel()
                        await forward_task  # 收集异常（如有）
                        break
            finally:
                # 轮次退出（含断连）时确保转发任务收尾
                if not forward_task.done():
                    forward_task.cancel()
    except WebSocketDisconnect:
        pass
