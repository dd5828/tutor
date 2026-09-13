"""WebSocket /chat：收消息 → orchestrator → 逐事件 NDJSON 推回。

客户端协议：
- 开启新轮次：发单行 JSON（message 必填，session_id / capability 可选）
- 轮次进行中收到 wait_for_input 事件后，直接发**纯文本**作为回复
- 轮次进行中发送的任何消息都会被当作反问回复投递
- 轮次结束与下一轮消息同时到达时，该消息转存 carry 作下一轮首条（不丢失）
- 服务端每个事件回一行 NDJSON，done 事件标志本轮结束
"""

from __future__ import annotations

import asyncio
import contextlib
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
    carry: str | None = None  # 轮次边界抢收到下一轮首条消息时的暂存
    try:
        while True:
            # 等待开启新轮次的消息（轮次边界抢收的消息优先消费）
            raw = carry if carry is not None else await websocket.receive_text()
            carry = None
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
                # 轮次进行中：后续消息全部作为反问回复投递；轮次结束即停止接收。
                # 轮次结束与下一轮消息同时到达时，该消息转存 carry 交给外层
                # 当下一轮首条（不丢消息）。
                while True:
                    recv_task = asyncio.create_task(websocket.receive_text())
                    done, _ = await asyncio.wait(
                        {recv_task, forward_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if forward_task in done:
                        # 先收尾再退出：recv_task 已取到的消息转存 carry
                        if recv_task in done:
                            try:
                                carry = recv_task.result()
                            except (WebSocketDisconnect, RuntimeError):
                                carry = None
                        # cancel 后必须 await 到位：外层紧接着的 receive_text
                        # 不能在旧接收协程还挂着时启动（否则 websockets 抛
                        # "cannot call recv while another coroutine is waiting"）
                        recv_task.cancel()
                        with contextlib.suppress(
                            asyncio.CancelledError, WebSocketDisconnect, RuntimeError
                        ):
                            await recv_task
                        await forward_task  # 收集异常（如有）
                        break
                    try:
                        reply_queue.put_nowait(recv_task.result())
                    except WebSocketDisconnect:
                        forward_task.cancel()
                        break
            finally:
                # 轮次退出（含断连）时确保转发任务收尾
                if not forward_task.done():
                    forward_task.cancel()
    except WebSocketDisconnect:
        pass
