"""chat 能力：默认对话（v1 单轮 LLM 调用，无工具循环）。

agent_loop（LLM↔工具多轮）Phase 1 起加。
"""

from __future__ import annotations

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext
from nnnu.services.llm import LLMClient

from .prompts import SYSTEM_PROMPT


class ChatCapability(BaseCapability):
    """默认对话：系统提示词 + 历史 + 本轮输入 → LLM 流式回复。"""

    manifest = CapabilityManifest(
        name="chat",
        description="默认对话：教育辅导人设的流式问答",
        stages=["回复"],
        tools_used=[],
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        # 测试可注入假 LLMClient；生产用环境变量构造。
        self._llm = llm if llm is not None else LLMClient()

    def _build_messages(self, ctx: TurnContext) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(ctx.conversation_history)
        messages.append({"role": "user", "content": ctx.user_message})
        return messages

    async def run(self, ctx: TurnContext, bus: StreamBus) -> None:
        async with bus.stage("回复", source=self.name):
            async for chunk in self._llm.chat_stream(self._build_messages(ctx)):
                await bus.content(chunk, source=self.name)
