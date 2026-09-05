"""Stream 事件协议（参考 DeepTutor core/stream.py，v1 精简版）。

所有能力/工具通过统一事件格式向消费者（WebSocket 推送、前端渲染）汇报进度。
v1 自上游 15 种事件保留 10 种：砍掉 thinking / observation / progress /
session / session_meta（见 docs/upstream-notes.md §2.1）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StreamEventType(StrEnum):
    """一个流式会话中所有可能的事件类型。"""

    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SOURCES = "sources"
    RESULT = "result"
    ERROR = "error"
    DONE = "done"
    WAIT_FOR_INPUT = "wait_for_input"


@dataclass
class StreamEvent:
    """一轮对话中发出的单个流式事件。

    Attributes:
        type: 事件语义类型。
        source: 产生事件的能力/工具名（如 "chat"、"orchestrator"）。
        stage: 当前所处阶段名（如 "引导"）。
        content: 人类可读文本载荷。
        metadata: 结构化数据（工具参数、引用来源、指标等）。
        session_id: 会话标识（由 orchestrator 填充）。
        turn_id: 轮次标识。
        seq: 事件序号（v1 预留，暂不自动分配）。
        timestamp: 事件创建时刻（Unix 秒）。
    """

    type: StreamEventType
    source: str = ""
    stage: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    turn_id: str = ""
    seq: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 编码的 dict（type 输出字符串枚举值）。"""
        return {
            "type": self.type.value,
            "source": self.source,
            "stage": self.stage,
            "content": self.content,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
        }
