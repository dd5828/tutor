"""math 能力：考研数学（高等数学）引导式教学（教学法文档 §4 的实现）。

教学循环四段：
1. 诊断：复述确认题意 → wait_for_input 问卡点 → 估水平档位（存 ctx.metadata["level"]）
2. 引导：一步一提示 + wait_for_input 等作答 + 结构化判答推进/纠错，
   连续失败降级提示粒度（§4.2 降级阶梯），答对解出即收；
3. 讲解：四段结构（考点/思路/关键步骤/易错点），回扣错因；已解出则收尾总结；
4. 巩固：变式题 + 一句话方法总结。

v1 约定：ctx.user_message 即题目（学生带题来）；AI 出题练走 quiz 能力。
判答/估档位走 pedagogy.py（签名固定，Phase 2 可换判题工具）。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo, UsageTracker, get_usage_tracker

from . import pedagogy
from .prompts import (
    CONSOLIDATE_PROMPT,
    DIAGNOSE_PROMPT,
    EXPLAIN_INSTRUCTION_SOLVED,
    EXPLAIN_INSTRUCTION_STUCK,
    EXPLAIN_PROMPT,
    HINT_PROMPT,
    SYSTEM_PROMPT,
)


class _UsageAccumulator:
    """把一轮中多次 LLM 调用的 token 用量加总，轮末一次记日志。"""

    def __init__(self) -> None:
        self._prompt = 0
        self._completion = 0
        self._total = 0
        self._count = 0

    def add(self, usage: UsageInfo) -> None:
        self._prompt += usage.prompt_tokens
        self._completion += usage.completion_tokens
        self._total += usage.total_tokens
        self._count += 1

    def usage(self) -> UsageInfo | None:
        """返回加总用量；一轮中没有任何用量尾块时返回 None。"""
        if self._count == 0:
            return None
        return UsageInfo(self._prompt, self._completion, self._total)


class MathTutorCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="math",
        description="考研数学（高等数学）陪练：诊断卡点→逐步引导→讲解→变式巩固",
        stages=["诊断", "引导", "讲解", "巩固"],
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

    async def _stream(
        self,
        user_prompt: str,
        bus: StreamBus,
        *,
        on_usage: Callable[[UsageInfo], None] | None = None,
    ) -> str:
        """带系统提示词的流式调用：逐段转发到总线，返回完整文本（判答要用）。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        chunks: list[str] = []
        async for chunk in self._llm.chat_stream(messages, on_usage=on_usage):
            chunks.append(chunk)
            await bus.content(chunk, source=self.name)
        return "".join(chunks)

    async def run(self, ctx: TurnContext, bus: StreamBus) -> None:
        question = ctx.user_message.strip()
        started = time.monotonic()
        usage_acc = _UsageAccumulator()

        try:
            # ---- 诊断：复述题意 + 问卡点 + 估水平档位 ----
            async with bus.stage("诊断", source=self.name):
                await self._stream(
                    DIAGNOSE_PROMPT.format(question=question),
                    bus,
                    on_usage=usage_acc.add,
                )
                stuck = (
                    await bus.wait_for_input("你卡在哪一步？算到哪了？", source=self.name)
                ).strip()
                level = await pedagogy.estimate_level(
                    self._llm, question=question, reply=stuck, on_usage=usage_acc.add
                )
                ctx.metadata["level"] = level

            # ---- 引导：一步一提示 + 判答，连续失败降级提示粒度 ----
            error_layer = pedagogy.NO_ERROR_LAYER
            transcript: list[str] = []
            hint_level = 0
            failures = 0
            solved = False
            async with bus.stage("引导", source=self.name):
                for _ in range(pedagogy.MAX_GUIDE_ROUNDS):
                    hint = await self._stream(
                        HINT_PROMPT.format(
                            question=question,
                            stuck_point=stuck,
                            level=level,
                            hint_level_name=pedagogy.HINT_LEVELS[hint_level],
                            hint_level_rule=pedagogy.HINT_LEVEL_RULES[hint_level],
                            error_layer=error_layer or "无",
                            transcript="\n".join(transcript) or "（第一轮，无历史）",
                        ),
                        bus,
                        on_usage=usage_acc.add,
                    )
                    reply = (
                        await bus.wait_for_input("把你的答案或思路发我", source=self.name)
                    ).strip()
                    if not reply:
                        break  # 空输入（超时/断连）：不再追问，直接讲解
                    judgement = await pedagogy.judge_answer(
                        self._llm,
                        question=question,
                        last_hint=hint,
                        reply=reply,
                        on_usage=usage_acc.add,
                    )
                    transcript.append(
                        f"提示（{pedagogy.HINT_LEVELS[hint_level]}）：{hint}\n"
                        f"学生答：{reply}\n"
                        f"判断：{'对' if judgement.correct else '错'}，"
                        f"错因 {judgement.error_layer or '无'}，{judgement.next_step}"
                    )
                    if judgement.correct:
                        failures = 0
                        if judgement.next_step == pedagogy.NEXT_STEP_SOLVED:
                            solved = True
                            break
                        continue
                    failures += 1
                    error_layer = judgement.error_layer or error_layer
                    if failures >= pedagogy.FAILURES_BEFORE_FINER_HINT:
                        hint_level = min(hint_level + 1, len(pedagogy.HINT_LEVELS) - 1)
                    if (
                        failures >= pedagogy.MAX_FAILURES
                        or judgement.next_step == pedagogy.NEXT_STEP_EXPLAIN
                    ):
                        break

            # ---- 讲解：四段结构（已解出则收尾总结），回扣错因 ----
            async with bus.stage("讲解", source=self.name):
                await self._stream(
                    EXPLAIN_PROMPT.format(
                        question=question,
                        stuck_point=stuck,
                        error_layer=error_layer or "无",
                        transcript="\n".join(transcript) or "（无引导历史）",
                        explain_instruction=(
                            EXPLAIN_INSTRUCTION_SOLVED
                            if solved
                            else EXPLAIN_INSTRUCTION_STUCK
                        ),
                    ),
                    bus,
                    on_usage=usage_acc.add,
                )

            # ---- 巩固：变式题 + 方法总结 ----
            async with bus.stage("巩固", source=self.name):
                await self._stream(
                    CONSOLIDATE_PROMPT.format(question=question),
                    bus,
                    on_usage=usage_acc.add,
                )
        finally:
            # 轮末统一记用量；异常中断也记，保证成本可追踪（风险表"成本失控"）
            final_usage = usage_acc.usage()
            if final_usage is not None:
                self._usage.record(
                    session_id=ctx.session_id,
                    turn_id=str(ctx.metadata.get("turn_id", "")),
                    capability=self.name,
                    usage=final_usage,
                    duration=time.monotonic() - started,
                )
