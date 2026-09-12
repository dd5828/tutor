"""考研数学教学法辅助逻辑：判答与水平估计（LLM 结构化输出 + 保守降级）。

教学质量不稳定是产品最大风险：本模块把所有"LLM 输出 JSON"统一收敛到
_structured_call + extract_json，解析失败一律走保守策略（宁可多讲一遍，
不在错误的判断上继续推进）。Phase 2 引入判题工具后，judge_answer /
estimate_level 换内部实现、签名不变，stage 流程不动。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo

from .prompts import JUDGE_JSON_PROMPT, LEVEL_JSON_PROMPT

# 难度三档（教学法文档 §3）：基础=单考点，常规=多考点组合，提高=综合技巧
LEVELS = ("基础", "常规", "提高")
DEFAULT_LEVEL = "常规"

# 错因四层（教学法文档 §5）
ERROR_LAYERS = ("概念", "公式", "方法", "计算")
NO_ERROR_LAYER = ""

# 引导节奏（教学法文档 §4.2 降级阶梯）
HINT_LEVELS = ("最小提示", "更细提示")
HINT_LEVEL_RULES = (
    "只指出考点或提醒题目条件，不给结果",
    "可给出中间结论或具体公式，但仍不给完整步骤",
)
MAX_GUIDE_ROUNDS = 4  # 引导最多反问轮数（防无限循环烧 token）
FAILURES_BEFORE_FINER_HINT = 2  # 连续失败几次后降级到更细提示
MAX_FAILURES = 3  # 连续失败上限，超过进入讲解

# 判答 next_step 取值（教学法文档 §7.3）
NEXT_STEP_CONTINUE = "continue"  # 有进展，继续引导
NEXT_STEP_SOLVED = "solved"  # 已解出本题，进入讲解总结
NEXT_STEP_EXPLAIN = "explain"  # 引导失败/主动要求，进入完整讲解


@dataclass
class Judgement:
    """一次判答结果。"""

    correct: bool
    error_layer: str
    next_step: str


def extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中提取 JSON 对象：容忍代码围栏与前后杂字，失败返回 None。

    评测运行器（tests/evals/run_evals.py）判分复用此函数。
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


async def _structured_call(
    llm: LLMClient,
    user_prompt: str,
    *,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> str:
    """调 LLM 要求只输出 JSON（低温提高稳定性），累积整段文本返回。"""
    chunks: list[str] = []
    async for chunk in llm.chat_stream(
        [{"role": "user", "content": user_prompt}],
        temperature=0.0,
        on_usage=on_usage,
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def estimate_level(
    llm: LLMClient,
    *,
    question: str,
    reply: str,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> str:
    """估计学生水平档位；解析失败回退常规档（不因解析错误走极端粒度）。"""
    text = await _structured_call(
        llm, LEVEL_JSON_PROMPT.format(question=question, reply=reply), on_usage=on_usage
    )
    data = extract_json(text) or {}
    level = str(data.get("level", ""))
    return level if level in LEVELS else DEFAULT_LEVEL


async def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    last_hint: str,
    reply: str,
    on_usage: Callable[[UsageInfo], None] | None = None,
) -> Judgement:
    """判答：对错 + 错因层 + 下一步（教学法文档 §5 / §7.3）。

    解析失败保守降级为"未解出、进入讲解"：宁可多讲一遍，也不要在错误的
    correct 判断上继续推进。
    """
    text = await _structured_call(
        llm,
        JUDGE_JSON_PROMPT.format(question=question, last_hint=last_hint, reply=reply),
        on_usage=on_usage,
    )
    data = extract_json(text) or {}
    error_layer = str(data.get("error_layer", ""))
    if error_layer not in ERROR_LAYERS:
        error_layer = NO_ERROR_LAYER
    next_step = str(data.get("next_step", ""))
    if next_step not in (NEXT_STEP_CONTINUE, NEXT_STEP_SOLVED, NEXT_STEP_EXPLAIN):
        next_step = NEXT_STEP_EXPLAIN
    return Judgement(
        correct=bool(data.get("correct")),
        error_layer=error_layer,
        next_step=next_step,
    )
