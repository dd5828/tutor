"""echo 假能力：把用户输入原样回显。

用途：验证 WS → orchestrator → 能力 全链路（不依赖 LLM，测试与演示均可）。
"""

from __future__ import annotations

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext


class EchoCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="echo",
        description="回显用户输入（全链路测试用）",
        stages=["回显"],
        tools_used=[],
    )

    async def run(self, ctx: TurnContext, bus: StreamBus) -> None:
        async with bus.stage("回显", source=self.name):
            await bus.content(f"echo: {ctx.user_message}", source=self.name)
