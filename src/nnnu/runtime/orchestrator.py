"""Orchestrator 统一入口（v1 精简版）。

所有入口（WebSocket / CLI）都调 handle()：补 session_id → 查能力注册表路由 →
后台任务跑能力、主协程 yield 每个事件（并发结构是流式输出的关键）。

v1 未做：PageIndex 校验、SESSION 元信息事件（会话元信息走 REST）、按 turn_id
的总线注册表、全局事件总线发布。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from nnnu.core import StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.runtime.registry import CapabilityRegistry, get_capability_registry

logger = logging.getLogger(__name__)


def _reflected_error_metadata(exc: Exception, status: str) -> dict[str, Any]:
    """按反射约定（见 core/errors.py docstring）提取异常属性。

    orchestrator 不 import ChatError，只用 getattr 取值：属性名是跨模块约定，
    改名即破坏反射。
    """
    metadata: dict[str, Any] = {"turn_terminal": True, "status": status}
    error_code = getattr(exc, "error_code", None)
    if isinstance(error_code, str) and error_code:
        metadata["error_code"] = error_code
    retryable = getattr(exc, "retryable", None)
    if isinstance(retryable, bool):
        metadata["retryable"] = retryable
    partial_response = getattr(exc, "partial_response", None)
    if isinstance(partial_response, str) and partial_response:
        metadata["partial_response"] = partial_response
    return metadata


class Orchestrator:
    """统一入口：把一轮对话路由到对应能力并流式转发事件。"""

    def __init__(self, caps: CapabilityRegistry | None = None) -> None:
        # 默认挂全局注册表；测试可注入独立实例。
        self._caps = caps if caps is not None else get_capability_registry()

    async def handle(self, ctx: TurnContext) -> AsyncIterator[StreamEvent]:
        """处理一轮对话，yield 流式事件。

        ``ctx.active_capability`` 指定能力；None 走默认 chat 能力。
        会话首次请求（session_id 为空）自动补发 UUID。
        """
        if not ctx.session_id:
            ctx.session_id = str(uuid.uuid4())

        cap_name = ctx.active_capability or "chat"
        capability = self._caps.get(cap_name)
        if capability is None:
            bus = StreamBus()
            await bus.error(
                f"未知能力: {cap_name}。可用: {self._caps.list_capabilities()}",
                source="orchestrator",
                metadata={"turn_terminal": True, "status": "failed"},
            )
            await bus.emit(
                StreamEvent(
                    type=StreamEventType.DONE,
                    source="orchestrator",
                    metadata={"status": "failed"},
                )
            )
            await bus.close()
            async for event in bus.subscribe():
                yield event
            return

        bus = StreamBus()

        async def _run() -> None:
            status = "completed"
            try:
                await capability.run(ctx, bus)
            except Exception as exc:
                status = "failed"
                logger.error("能力 %s 执行失败: %s", cap_name, exc, exc_info=True)
                await bus.error(
                    str(exc),
                    source=cap_name,
                    metadata=_reflected_error_metadata(exc, status),
                )
            finally:
                await bus.emit(
                    StreamEvent(
                        type=StreamEventType.DONE,
                        source=cap_name,
                        metadata={"status": status},
                    )
                )
                await bus.close()

        task = asyncio.create_task(_run())
        async for event in bus.subscribe():
            yield event
        await task
