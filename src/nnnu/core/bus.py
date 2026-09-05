"""StreamBus 流式事件总线（参考 DeepTutor core/stream_bus.py，v1 抄约 70%）。

能力/工具向总线 emit，消费者（WebSocket 推送）subscribe 读取。
v1 砍掉：register_bus/unregister_bus/get_bus 按 turn_id 的全局总线表（wait_for_input
已覆盖反问需求）；thinking/observation/progress 便捷方法；trace.py 元数据合并
（简化为本地 _merge_metadata）。抄写细节见 docs/upstream-notes.md §2.3。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from .events import StreamEvent, StreamEventType


def _merge_metadata(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """合并事件 metadata：extra（调用方）覆盖 base（协议字段），
    与上游 merge_trace_metadata 同序。"""
    merged = dict(base)
    if extra:
        merged.update(extra)
    return merged


class StreamBus:
    """单轮对话的扇出式异步事件总线。"""

    def __init__(self, *, max_history: int | None = None) -> None:
        self._subscribers: list[asyncio.Queue[StreamEvent | None]] = []
        self._closed = False
        self._history: list[StreamEvent] = []
        self._input_listeners: list[asyncio.Queue[str]] = []
        # None = 保留全部历史（单轮总线随轮次生灭）；长期存活的传上限避免无限增长。
        self._max_history = max_history

    async def emit(self, event: StreamEvent) -> None:
        """向每个活跃订阅者推送 *event*。总线关闭后的事件被丢弃。"""
        if self._closed:
            return
        self._history.append(event)
        if self._max_history is not None and len(self._history) > self._max_history:
            del self._history[: len(self._history) - self._max_history]
        for q in self._subscribers:
            await q.put(event)

    async def subscribe(self) -> AsyncIterator[StreamEvent]:
        """yield 事件直到总线关闭（先重放历史，再实时接收）。"""
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._subscribers.append(q)
        # 快照重放范围与队列注册在同一同步步骤完成：重放期间新 emit 的事件
        # 只经队列送达；若遍历活列表则会在重放时把同一事件 yield 两次。
        replay_count = len(self._history)
        try:
            for event in self._history[:replay_count]:
                yield event
            if self._closed and q.empty():
                return
            while True:
                queued = await q.get()
                if queued is None:
                    break
                yield queued
        finally:
            self._subscribers.remove(q)

    def mark_closed(self) -> None:
        """同步版 close()。

        订阅队列无界，put_nowait 投递终止哨兵永不阻塞，因此同步调用点
        （引擎 teardown、CLI 路径）无需事件循环即可关停总线。
        """
        self._closed = True
        for q in self._subscribers:
            q.put_nowait(None)

    async def close(self) -> None:
        """通知所有订阅者流已结束。"""
        self.mark_closed()

    # ---- 生产者便捷方法 ----

    @asynccontextmanager
    async def stage(
        self,
        name: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[None]:
        """在代码块前后发出 STAGE_START / STAGE_END（块内异常也保证配对）。"""
        meta = metadata or {}
        await self.emit(
            StreamEvent(
                type=StreamEventType.STAGE_START, source=source, stage=name, metadata=meta
            )
        )
        try:
            yield
        finally:
            await self.emit(
                StreamEvent(
                    type=StreamEventType.STAGE_END, source=source, stage=name, metadata=meta
                )
            )

    async def content(
        self,
        text: str,
        source: str = "",
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.CONTENT,
                source=source,
                stage=stage,
                content=text,
                metadata=metadata or {},
            )
        )

    async def tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        source: str = "",
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.TOOL_CALL,
                source=source,
                stage=stage,
                content=tool_name,
                metadata=_merge_metadata({"args": args}, metadata),
            )
        )

    async def tool_result(
        self,
        tool_name: str,
        result: str,
        source: str = "",
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                source=source,
                stage=stage,
                content=result,
                metadata=_merge_metadata({"tool": tool_name}, metadata),
            )
        )

    async def sources(
        self,
        sources: list[dict[str, Any]],
        source: str = "",
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.SOURCES,
                source=source,
                stage=stage,
                metadata=_merge_metadata({"sources": sources}, metadata),
            )
        )

    async def result(
        self,
        data: dict[str, Any],
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.RESULT,
                source=source,
                metadata=_merge_metadata(data, metadata),
            )
        )

    async def error(
        self,
        message: str,
        source: str = "",
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            StreamEvent(
                type=StreamEventType.ERROR,
                source=source,
                stage=stage,
                content=message,
                metadata=metadata or {},
            )
        )

    async def wait_for_input(
        self,
        prompt: str,
        source: str = "",
        stage: str = "",
        timeout: float | None = None,
    ) -> str:
        """暂停能力执行，等待前端回传用户输入。

        Returns:
            用户输入；*timeout* 秒内无输入返回空串（如 CLI 等无法回传输入的入口）。
            timeout=None（默认）对交互客户端无限等待。
        """
        await self.emit(
            StreamEvent(
                type=StreamEventType.WAIT_FOR_INPUT, source=source, stage=stage, content=prompt
            )
        )
        input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._input_listeners.append(input_queue)
        try:
            return await asyncio.wait_for(input_queue.get(), timeout=timeout)
        except TimeoutError:
            return ""
        finally:
            if input_queue in self._input_listeners:
                self._input_listeners.remove(input_queue)

    def submit_input(self, content: str) -> None:
        """接收前端/WS 回传的用户输入，投递给所有等待者。"""
        for q in self._input_listeners:
            q.put_nowait(content)
        self._input_listeners.clear()

    # ---- 消费者适配 ----

    @staticmethod
    def event_to_json(event: StreamEvent) -> str:
        """序列化为单行 JSON（NDJSON）。ensure_ascii=False：中文场景关键。"""
        return json.dumps(event.to_dict(), ensure_ascii=False)
