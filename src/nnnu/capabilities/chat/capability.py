"""chat 能力：默认对话（v1 单轮 LLM 调用，无工具循环）。

agent_loop（LLM↔工具多轮）Phase 1 起加。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo, UsageTracker, get_usage_tracker

from .prompts import SYSTEM_PROMPT


class ChatCapability(BaseCapability):
    """默认对话：系统提示词 + 历史 + 本轮输入 → LLM 流式回复。

    流结束时经 on_usage 回调把 token 用量交给 UsageTracker 记日志。
    """

    manifest = CapabilityManifest(
        name="chat",
        description="默认对话：教育辅导人设的流式问答",
        stages=["回复"],
        tools_used=[],
    )

    def __init__(
        self,
        llm: LLMClient | None = None,
        usage: UsageTracker | None = None,
    ) -> None:
        # 测试可注入假 LLMClient / 假 tracker；生产用环境变量与全局单例。
        self._llm = llm if llm is not None else LLMClient()
        self._usage = usage if usage is not None else get_usage_tracker()

    def _build_messages(self, ctx: TurnContext) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(ctx.conversation_history)
        messages.append({"role": "user", "content": ctx.user_message})
        return messages

    def _on_usage(self, ctx: TurnContext, started: float) -> Callable[[UsageInfo], None]:
        def record(usage: UsageInfo) -> None:
            self._usage.record(
                session_id=ctx.session_id,
                turn_id=str(ctx.metadata.get("turn_id", "")),
                capability=self.name,
                usage=usage,
                duration=time.monotonic() - started,
            )

        return record

    async def run(self, ctx: TurnContext, bus: StreamBus) -> None:
        started = time.monotonic()
        async with bus.stage("回复", source=self.name):
            async for chunk in self._llm.chat_stream(
                self._build_messages(ctx),
                on_usage=self._on_usage(ctx, started),
            ):
                await bus.content(chunk, source=self.name)
