"""math 能力：考研数学（高等数学）引导式教学（教学法文档 §4 的实现）。

教学循环四段：
1. 诊断：复述确认题意 → wait_for_input 问卡点 → 估水平档位（存 ctx.metadata["level"]）
2. 引导：一步一提示 + wait_for_input 等作答 + 结构化判答推进/纠错，
   连续失败降级提示粒度（§4.2 降级阶梯），答对解出即收；
3. 讲解：四段结构（考点/思路/关键步骤/易错点），回扣错因；已解出则收尾总结；
4. 巩固：变式题 + 一句话方法总结。

v1 约定：ctx.user_message 即题目（学生带题来）；AI 出题练走 quiz 能力。
判答/估档位走 pedagogy.py（签名固定，Phase 2 可换判题工具）。
讲解阶段经 rag 工具检索知识库（_collect_references），无工具/无索引时
静默降级，不新增 LLM 调用。
W6 起：引导判答经 check_answer 工具（_judge_via_tool）、巩固变式题经
question_bank 工具（_collect_consolidation_question）——两者在工具缺失时
都立即短路回退原实现，无工具路径提示词与调用数逐字节不变。
W7 起：轮首经 read_memory 工具读画像（_load_profile）注入系统提示词与
诊断点名，轮末经 write_memory 工具写回（_write_memory）——记忆故障一律
静默降级，无工具/无画像路径提示词与调用数逐字节不变。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nnnu.core import BaseCapability, CapabilityManifest, StreamBus, TurnContext
from nnnu.runtime.registry import ToolRegistry, get_tool_registry
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo, UsageTracker, get_usage_tracker

from . import pedagogy
from .prompts import (
    CONSOLIDATE_BANK_PROMPT,
    CONSOLIDATE_PROMPT,
    DIAGNOSE_PROMPT,
    DIAGNOSE_WITH_PROFILE,
    EXPLAIN_INSTRUCTION_SOLVED,
    EXPLAIN_INSTRUCTION_STUCK,
    EXPLAIN_PROMPT,
    HINT_PROMPT,
    PROFILE_PROMPT,
    REFERENCES_PROMPT,
    SYSTEM_PROMPT,
    TOPIC_JSON_PROMPT,
)
from .topics import TOPICS


async def _collect_references(
    tools: ToolRegistry | None, question: str
) -> tuple[str, list[dict[str, Any]]]:
    """查 rag 工具取讲解参考资料；无工具/无索引/无命中/异常一律返回 ("", [])。

    教学主流程不因知识库问题中断（完全静默降级）。换检索工具只改本函数
    内部，不动 stage 流程。
    """
    if tools is None:
        return "", []
    tool = tools.get("rag")
    if tool is None:
        return "", []
    try:
        result = await tool.execute(query=question)
    except Exception:
        return "", []
    if not result.success or not result.sources:
        return "", []
    refs = "\n".join(_format_reference(src, i) for i, src in enumerate(result.sources, 1))
    return refs, result.sources


def _format_reference(src: dict[str, Any], index: int) -> str:
    title = str(src.get("title") or "资料")
    try:
        score = float(src.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    excerpt = str(src.get("excerpt") or "")
    return f"[{index}] {title}（相关度 {score:.2f}）：{excerpt}"


async def _extract_topic(
    llm: LLMClient,
    question: str,
    *,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> str | None:
    """LLM 结构化提取考点（temperature=0）；解析失败/词表外返回 None。

    返回 None 时调用方保守回退 LLM 出题（不做第二次尝试）。
    """
    text = await pedagogy._structured_call(
        llm, TOPIC_JSON_PROMPT.format(question=question), on_usage=on_usage
    )
    data = pedagogy.extract_json(text) or {}
    topic = str(data.get("topic", ""))
    return topic if topic in TOPICS else None


@dataclass(frozen=True)
class MemorySnapshot:
    """轮首 read_memory 取回的画像视图（能力层只消费字符串协议面）。"""

    summary: str = ""
    last_topic: str = ""
    last_stuck: str = ""


async def _load_profile(
    tools: ToolRegistry | None, session_id: str
) -> MemorySnapshot | None:
    """轮首读画像；无工具/空 session_id/读取失败/摘要为空 → None。

    返回 None 即"无画像路径"：系统提示词与诊断模板逐字节不变。
    """
    if tools is None or not session_id:
        return None
    tool = tools.get("read_memory")
    if tool is None:
        return None
    try:
        result = await tool.execute(session_id=session_id)
    except Exception:
        return None
    if not result.success:
        return None
    summary = str(result.metadata.get("summary", "")).strip()
    if not summary:
        return None
    return MemorySnapshot(
        summary=summary,
        last_topic=str(result.metadata.get("last_topic", "")),
        last_stuck=str(result.metadata.get("last_stuck", "")),
    )


async def _write_memory(
    tools: ToolRegistry | None,
    *,
    session_id: str,
    question: str,
    level: str,
    topic: str | None,
    error_layer: str,
    solved: bool,
    stuck_point: str,
) -> None:
    """轮末写画像；无工具/空 session_id/写失败一律静默（记忆故障不打断教学）。"""
    if tools is None or not session_id:
        return
    tool = tools.get("write_memory")
    if tool is None:
        return
    try:
        await tool.execute(
            session_id=session_id,
            question=question,
            level=level,
            topic=topic or "",
            error_layer=error_layer,
            solved=solved,
            stuck_point=stuck_point,
        )
    except Exception:
        return


async def _ensure_topic(
    tools: ToolRegistry | None,
    llm: LLMClient,
    *,
    question: str,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> str | None:
    """每轮至多一次话题提取；question_bank 工具不存在时绝不发起 LLM 调用。

    提取结果由调用方缓存进 ctx.metadata["topic"]，巩固查库与轮末写回共用
    （硬守卫：memory 工具的存在不会导致新增提取调用）。
    """
    if tools is None or tools.get("question_bank") is None:
        return None
    return await _extract_topic(llm, question=question, on_usage=on_usage)


async def _collect_consolidation_question(
    tools: ToolRegistry | None,
    *,
    level: str,
    topic: str | None,
) -> tuple[str, dict[str, Any] | None]:
    """查 question_bank 取巩固变式题，返回 (变式题文本, bank item)。

    topic 由调用方（_ensure_topic 缓存）决定；无工具/无 question_bank
    工具/topic 为空/查库未命中/工具异常 → ("", None)，调用方走既有
    CONSOLIDATE_PROMPT 路径（提示词逐字节不变）。本函数不发起 LLM 调用。
    """
    if tools is None or topic is None:
        return "", None
    tool = tools.get("question_bank")
    if tool is None:
        return "", None
    try:
        result = await tool.execute(topic=topic, level=level)
    except Exception:
        return "", None
    if not result.success:
        return "", None
    item = result.metadata.get("item")
    if not isinstance(item, dict) or not str(item.get("question", "")).strip():
        return "", None
    return str(item["question"]), item


async def _judge_via_tool(
    tools: ToolRegistry | None,
    llm: LLMClient,
    *,
    question: str,
    last_hint: str,
    reply: str,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> pedagogy.Judgement | None:
    """经 check_answer 工具判答；无工具/失败/结果不可解析返回 None。

    返回 None 时调用方回退 pedagogy.judge_answer 直调（提示词、
    temperature、调用次数、usage 全不变）。metadata 还原时重做词表
    校验（镜像 judge_answer 的保守策略）。
    """
    if tools is None:
        return None
    tool = tools.get("check_answer")
    if tool is None:
        return None
    result = await tool.execute(
        question=question, last_hint=last_hint, reply=reply, on_usage=on_usage
    )
    if not result.success:
        return None
    data = result.metadata.get("judgement")
    if not isinstance(data, dict):
        return None
    error_layer = str(data.get("error_layer", ""))
    if error_layer not in pedagogy.ERROR_LAYERS:
        error_layer = pedagogy.NO_ERROR_LAYER
    next_step = str(data.get("next_step", ""))
    if next_step not in (
        pedagogy.NEXT_STEP_CONTINUE,
        pedagogy.NEXT_STEP_SOLVED,
        pedagogy.NEXT_STEP_EXPLAIN,
    ):
        next_step = pedagogy.NEXT_STEP_EXPLAIN
    return pedagogy.Judgement(
        correct=bool(data.get("correct")),
        error_layer=error_layer,
        next_step=next_step,
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
        tools_used=["rag", "check_answer", "question_bank", "read_memory", "write_memory"],
    )

    def __init__(
        self,
        llm: LLMClient | None = None,
        usage: UsageTracker | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        # 测试可注入假 LLMClient / 假 tracker / 假注册表；生产用环境变量与
        # 全局单例。无 rag 工具（未注册或索引未构建）时讲解阶段静默降级。
        self._llm = llm if llm is not None else LLMClient()
        self._usage = usage if usage is not None else get_usage_tracker()
        self._tools = tools if tools is not None else get_tool_registry()

    async def _stream(
        self,
        user_prompt: str,
        bus: StreamBus,
        *,
        profile_summary: str = "",
        on_usage: Callable[[UsageInfo], None] | None = None,
    ) -> str:
        """带系统提示词的流式调用：逐段转发到总线，返回完整文本（判答要用）。

        profile_summary 非空时把画像摘要拼接进系统提示词；为空时组装与
        无画像路径逐字节一致。判答/估档位/话题提取走 pedagogy 结构化
        调用（单 user 消息），不受画像注入影响。
        """
        system = SYSTEM_PROMPT
        if profile_summary:
            system = f"{SYSTEM_PROMPT}\n\n{PROFILE_PROMPT.format(summary=profile_summary)}"
        messages = [
            {"role": "system", "content": system},
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
        # 轮末写回所需：try 前初始化（异常轮 finally 安全读取，不掩盖原异常）
        stuck = ""
        level = pedagogy.DEFAULT_LEVEL
        error_layer = pedagogy.NO_ERROR_LAYER
        solved = False
        topic: str | None = None

        try:
            # ---- 轮首：读画像（失败静默），有画像则注入 ----
            memory = await _load_profile(self._tools, ctx.session_id)
            profile_summary_text = memory.summary if memory is not None else ""
            if profile_summary_text:
                ctx.student_profile = profile_summary_text

            # ---- 诊断：复述题意 + 问卡点 + 估水平档位 ----
            async with bus.stage("诊断", source=self.name):
                if memory is not None and memory.last_stuck:
                    diagnose_prompt = DIAGNOSE_WITH_PROFILE.format(
                        question=question,
                        last_topic=memory.last_topic or "相关考点",
                        last_stuck=memory.last_stuck,
                    )
                else:
                    # 无画像/无上次卡点：原模板逐字节不变
                    diagnose_prompt = DIAGNOSE_PROMPT.format(question=question)
                await self._stream(
                    diagnose_prompt,
                    bus,
                    profile_summary=profile_summary_text,
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
            transcript: list[str] = []
            hint_level = 0
            failures = 0
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
                        profile_summary=profile_summary_text,
                        on_usage=usage_acc.add,
                    )
                    reply = (
                        await bus.wait_for_input("把你的答案或思路发我", source=self.name)
                    ).strip()
                    if not reply:
                        break  # 空输入（超时/断连）：不再追问，直接讲解
                    judgement = await _judge_via_tool(
                        self._tools,
                        self._llm,
                        question=question,
                        last_hint=hint,
                        reply=reply,
                        on_usage=usage_acc.add,
                    )
                    if judgement is None:
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
                references, sources = await _collect_references(self._tools, question)
                if sources:
                    await bus.sources(
                        sources, source=self.name, stage="讲解", metadata={"query": question}
                    )
                explain_prompt = EXPLAIN_PROMPT.format(
                    question=question,
                    stuck_point=stuck,
                    error_layer=error_layer or "无",
                    transcript="\n".join(transcript) or "（无引导历史）",
                    explain_instruction=(
                        EXPLAIN_INSTRUCTION_SOLVED
                        if solved
                        else EXPLAIN_INSTRUCTION_STUCK
                    ),
                )
                # 仅在检索有命中时拼接参考资料（无命中路径提示词与改造前逐字节一致）
                if references:
                    explain_prompt = (
                        f"{explain_prompt}\n\n{REFERENCES_PROMPT.format(references=references)}"
                    )
                await self._stream(
                    explain_prompt,
                    bus,
                    profile_summary=profile_summary_text,
                    on_usage=usage_acc.add,
                )

            # ---- 巩固：变式题 + 方法总结 ----
            async with bus.stage("巩固", source=self.name):
                if topic is None:
                    topic = await _ensure_topic(
                        self._tools, self._llm, question=question, on_usage=usage_acc.add
                    )
                    if topic is not None:
                        ctx.metadata["topic"] = topic  # 话题缓存：轮末写回共用
                variant, bank_item = await _collect_consolidation_question(
                    self._tools,
                    level=str(ctx.metadata.get("level", pedagogy.DEFAULT_LEVEL)),
                    topic=topic,
                )
                if bank_item is None:
                    await self._stream(
                        CONSOLIDATE_PROMPT.format(question=question),
                        bus,
                        profile_summary=profile_summary_text,
                        on_usage=usage_acc.add,
                    )
                else:
                    # 题库命中：题目直发总线，LLM 只写一句话总结（answer 不展示）
                    await bus.content(f"变式题：{variant}", source=self.name, stage="巩固")
                    await self._stream(
                        CONSOLIDATE_BANK_PROMPT.format(question=question),
                        bus,
                        profile_summary=profile_summary_text,
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
            # 轮末写画像（usage 记录之后）：记忆故障静默，异常轮也写（半成品数据 schema 宽容）
            await _write_memory(
                self._tools,
                session_id=ctx.session_id,
                question=question,
                level=str(ctx.metadata.get("level", pedagogy.DEFAULT_LEVEL)),
                topic=topic,
                error_layer=error_layer,
                solved=solved,
                stuck_point=stuck,
            )
