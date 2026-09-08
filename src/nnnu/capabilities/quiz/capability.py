"""quiz 演示能力：出题 → 反问等答案 → 批改引导。

展示教学循环的两个核心机制：stage 分段（前端进度条）与
wait_for_input 反问（能力暂停等学生作答，答案经传输层回路返回）。
"""

from __future__ import annotations

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext

QUESTIONS: list[tuple[str, str]] = [
    ("7 × 8 = ?", "56"),
    ("一个数除以它本身（非零）等于多少？", "1"),
]


class QuizCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="quiz",
        description="演示反问教学循环：出题→等答案→批改引导",
        stages=["出题", "批改"],
        tools_used=[],
    )

    def __init__(self) -> None:
        self._index = 0

    async def run(self, ctx: TurnContext, bus: StreamBus) -> None:
        question, answer = QUESTIONS[self._index % len(QUESTIONS)]
        self._index += 1

        async with bus.stage("出题", source=self.name):
            await bus.content(f"第 {self._index} 题：{question}", source=self.name)
        async with bus.stage("批改", source=self.name):
            reply = await bus.wait_for_input("请输入你的答案", source=self.name)
            if reply.strip() == answer:
                await bus.content("正确！", source=self.name)
            else:
                await bus.content(
                    f"答案是 {answer}，你答的是 {reply or '(未作答)'}。"
                    f"小提示：{question} 可以拆成更小的步骤想。",
                    source=self.name,
                )
