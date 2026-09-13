"""学习记忆工具（Phase 3 W7）：read_memory / write_memory。

两个工具同文件（loader+tool 同文件的先例）。**读写永不抛异常**：
记忆是教学的增强信息而非前置条件，一切失败降级为 success=False 或
"暂无历史"，教学主流程绝不因记忆故障中断（与 rag_tool 同款哲学）。

画像按 session_id 键控；目录默认 data/memory，环境变量 NNNU_MEMORY_DIR
可覆盖（评测/测试用临时目录隔离）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from nnnu.capabilities.math.pedagogy import DEFAULT_LEVEL, ERROR_LAYERS, LEVELS
from nnnu.core import BaseTool, ToolDefinition, ToolParameter, ToolResult
from nnnu.services.memory import DEFAULT_MEMORY_DIR
from nnnu.services.memory.profile import (
    StudentProfile,
    TurnRecord,
    apply_turn,
    profile_summary,
    to_dict,
)
from nnnu.services.memory.store import load_profile, profile_path, save_profile


def resolve_memory_dir() -> Path:
    """NNNU_MEMORY_DIR 环境变量优先，否则 data/memory。"""
    env = os.environ.get("NNNU_MEMORY_DIR")
    return Path(env) if env else DEFAULT_MEMORY_DIR


class ReadMemoryTool(BaseTool):
    """读学生画像：返回摘要与上次卡点（能力层轮首注入用）。"""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._memory_dir = memory_dir if memory_dir is not None else resolve_memory_dir()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_memory",
            description=(
                "读取学生的学习画像：练习轮数、薄弱考点、易错点与上次卡点。"
                "教学开始前调用，用于个性化提示粒度与讲解深度。"
            ),
            parameters=[
                ToolParameter(
                    name="session_id",
                    type="string",
                    required=True,
                    description="会话标识（前端页级 UUID）",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            session_id = str(kwargs.get("session_id") or "").strip()
            path = profile_path(self._memory_dir, session_id)
            if path is None:
                return ToolResult(
                    content="session_id 非法（需字母数字开头，1-128 字符）", success=False
                )
            if not path.is_file():
                return ToolResult(
                    content="暂无历史画像",
                    success=True,
                    metadata={
                        "exists": False,
                        "corrupt": False,
                        "summary": "",
                        "last_topic": "",
                        "last_stuck": "",
                        "last_level": "",
                        "turn_count": 0,
                    },
                )
            profile = load_profile(path)
            if profile is None:
                # 文件存在但读不出来 = 损坏：提示并当作无画像继续
                return ToolResult(
                    content="历史画像文件损坏，已忽略（将从零开始记录）",
                    success=True,
                    metadata={
                        "exists": True,
                        "corrupt": True,
                        "summary": "",
                        "last_topic": "",
                        "last_stuck": "",
                        "last_level": "",
                        "turn_count": 0,
                    },
                )
            summary = profile_summary(profile)
            return ToolResult(
                content=summary or "暂无有效画像记录",
                success=True,
                metadata={
                    "exists": True,
                    "corrupt": False,
                    "summary": summary,
                    "last_topic": profile.last_turn.topic,
                    "last_stuck": profile.last_turn.stuck_point,
                    "last_level": profile.last_turn.level,
                    "turn_count": profile.turn_count,
                    "profile": to_dict(profile),
                },
            )
        except Exception as exc:  # 记忆故障不打断教学：兜底不抛
            return ToolResult(content=f"读取画像失败: {exc}", success=False)


class WriteMemoryTool(BaseTool):
    """写一轮教学结果入画像（record_turn 语义）。"""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._memory_dir = memory_dir if memory_dir is not None else resolve_memory_dir()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_memory",
            description=(
                "把一轮教学结果写入学生画像：题目、考点、错因层、是否解出、"
                "水平档位与卡点。教学轮次结束时调用。"
            ),
            parameters=[
                ToolParameter(
                    name="session_id", type="string", required=True, description="会话标识"
                ),
                ToolParameter(
                    name="question", type="string", required=True, description="本轮题目"
                ),
                ToolParameter(
                    name="topic",
                    type="string",
                    required=False,
                    description="考点（话题提取结果，可空）",
                ),
                ToolParameter(
                    name="level",
                    type="string",
                    required=False,
                    enum=list(LEVELS),
                    description="本轮诊断的水平档位",
                ),
                ToolParameter(
                    name="error_layer",
                    type="string",
                    required=False,
                    enum=list(ERROR_LAYERS),
                    description="本轮错因层（无错因填空串）",
                ),
                ToolParameter(
                    name="solved",
                    type="boolean",
                    required=False,
                    description="本轮最终是否解出",
                ),
                ToolParameter(
                    name="stuck_point",
                    type="string",
                    required=False,
                    description="学生自述的卡点",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            session_id = str(kwargs.get("session_id") or "").strip()
            path = profile_path(self._memory_dir, session_id)
            if path is None:
                return ToolResult(
                    content="session_id 非法（需字母数字开头，1-128 字符）", success=False
                )
            question = str(kwargs.get("question") or "").strip()
            if not question:
                return ToolResult(content="question 不能为空", success=False)
            topic = str(kwargs.get("topic") or "").strip()
            level = str(kwargs.get("level") or DEFAULT_LEVEL).strip()
            if level not in LEVELS:
                level = DEFAULT_LEVEL
            error_layer = str(kwargs.get("error_layer") or "").strip()
            if error_layer not in ERROR_LAYERS:
                error_layer = ""
            solved = bool(kwargs.get("solved"))
            stuck_point = str(kwargs.get("stuck_point") or "").strip()

            profile = load_profile(path) or StudentProfile()
            apply_turn(
                profile,
                TurnRecord(
                    question=question,
                    topic=topic,
                    error_layer=error_layer,
                    solved=solved,
                    level=level,
                    stuck_point=stuck_point,
                    timestamp=time.time(),
                ),
            )
            save_profile(path, profile)
            return ToolResult(
                content=f"画像已更新（累计 {profile.turn_count} 轮）",
                success=True,
                metadata={
                    "turn_count": profile.turn_count,
                    "summary": profile_summary(profile),
                },
            )
        except Exception as exc:  # 写失败不打断教学
            return ToolResult(content=f"写入画像失败: {exc}", success=False)
