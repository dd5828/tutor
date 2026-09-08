"""每轮对话 token 用量记录（v1 雏形）。

LLM 流结束时把 usage 交给 UsageTracker，输出一条 JSON 结构化日志
（session_id / turn_id / capability / tokens / cost / duration），
成本看板（Phase 4）在此数据基础上做聚合。

单价（每百万 token，人民币）默认按 deepseek-chat 价，需按官网现行价核对；
环境变量 NNNU_LLM_PRICE_IN / NNNU_LLM_PRICE_OUT 可覆盖，cost() 每次调用时读取。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("nnnu.usage")

DEFAULT_PRICE_IN = 2.0  # 输入
DEFAULT_PRICE_OUT = 8.0  # 输出


def _env_price(name: str, default: float) -> float:
    """读单价环境变量；缺失或非法回退默认值。"""
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


@dataclass
class UsageInfo:
    """一次 LLM 调用的 token 统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> UsageInfo:
        """从 OpenAI 兼容协议的 usage 对象构造（缺失字段按 0）。"""
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    def cost(self) -> float:
        """按每百万 token 单价算成本（人民币）。"""
        price_in = _env_price("NNNU_LLM_PRICE_IN", DEFAULT_PRICE_IN)
        price_out = _env_price("NNNU_LLM_PRICE_OUT", DEFAULT_PRICE_OUT)
        return (
            self.prompt_tokens / 1_000_000 * price_in
            + self.completion_tokens / 1_000_000 * price_out
        )


class UsageTracker:
    """按轮次记录用量，输出 JSON 结构化日志行。"""

    def record(
        self,
        *,
        session_id: str,
        capability: str,
        usage: UsageInfo,
        duration: float,
        turn_id: str = "",
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "usage",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "capability": capability,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost": round(usage.cost(), 6),
                    "duration": round(duration, 3),
                },
                ensure_ascii=False,
            )
        )


_default_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    """返回全局用量记录器（首次调用创建）。"""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = UsageTracker()
    return _default_tracker
