"""学习者画像（Phase 3 W7）：结构化字段 + 纯函数更新 + 确定性摘要。

分层约定：本模块是 services 层，不依赖 capabilities 层的词表
（档位/错因的取值钳制在写端 tools/memory_tools.py 完成）；画像档案
长存，读端宽容解析（坏条目跳过、缺失字段取默认值），写端严格。

摘要生成必须确定性：排序显式指定键，不依赖 dict 迭代顺序——同一画像
两次摘要逐字相等（单测守护），避免提示词注入的随机抖动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_ERROR_RECORDS = 20  # 错题记录上限（最近 N 条）
SUMMARY_MAX_LEN = 200  # 摘要长度护栏（拼接进系统提示词，控制成本）


@dataclass
class TopicMastery:
    """某考点的掌握度。"""

    level: str = ""  # 该考点最近一次诊断档位
    correct_count: int = 0
    wrong_count: int = 0


@dataclass
class ErrorRecord:
    """一条错题记录（仅 error_layer 非空的轮次入录）。"""

    question: str = ""
    topic: str = ""  # 话题提取失败/未提取时为空
    error_layer: str = ""
    solved: bool = False
    timestamp: float = 0.0


@dataclass
class LastTurn:
    """上一轮概况（诊断点名的直接来源）。"""

    level: str = ""
    topic: str = ""
    stuck_point: str = ""


@dataclass
class StudentProfile:
    """一个学生的完整画像。"""

    session_id: str = ""
    turn_count: int = 0
    mastery: dict[str, TopicMastery] = field(default_factory=dict)
    error_records: list[ErrorRecord] = field(default_factory=list)
    error_layers: dict[str, int] = field(default_factory=dict)  # 错因层 → 累计次数
    last_turn: LastTurn = field(default_factory=LastTurn)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class TurnRecord:
    """一轮教学结果的入参（write_memory 的 record_turn 语义）。"""

    question: str
    topic: str = ""
    error_layer: str = ""
    solved: bool = False
    level: str = ""
    stuck_point: str = ""
    timestamp: float = 0.0


def apply_turn(profile: StudentProfile, turn: TurnRecord) -> None:
    """把一轮结果并入画像（原地更新）。"""
    if profile.created_at == 0.0:
        profile.created_at = turn.timestamp
    profile.updated_at = turn.timestamp
    profile.turn_count += 1
    if turn.topic:
        mastery = profile.mastery.get(turn.topic)
        if mastery is None:
            mastery = profile.mastery[turn.topic] = TopicMastery(level=turn.level)
        else:
            mastery.level = turn.level
        if turn.solved:
            mastery.correct_count += 1
        else:
            mastery.wrong_count += 1
    if turn.error_layer:
        profile.error_layers[turn.error_layer] = (
            profile.error_layers.get(turn.error_layer, 0) + 1
        )
        profile.error_records.append(
            ErrorRecord(
                question=turn.question,
                topic=turn.topic,
                error_layer=turn.error_layer,
                solved=turn.solved,
                timestamp=turn.timestamp,
            )
        )
        del profile.error_records[:-MAX_ERROR_RECORDS]
    profile.last_turn = LastTurn(level=turn.level, topic=turn.topic, stuck_point=turn.stuck_point)


def profile_summary(profile: StudentProfile) -> str:
    """确定性画像摘要；空画像（turn_count==0）返回空串。"""
    if profile.turn_count == 0:
        return ""
    parts: list[str] = [f"该学生已练习 {profile.turn_count} 轮"]
    last = profile.last_turn
    if last.stuck_point:
        parts.append(
            f"上次在「{last.topic or '未知考点'}」卡在「{_clip(last.stuck_point, 40)}」"
            f"（水平 {last.level or '常规'}）"
        )
    errors = sorted(profile.error_layers.items(), key=lambda kv: (-kv[1], kv[0]))
    if errors:
        parts.append("易错点：" + "、".join(f"{layer}×{count}" for layer, count in errors[:3]))
    weak = sorted(
        ((topic, m) for topic, m in profile.mastery.items() if m.wrong_count > 0),
        key=lambda kv: (-kv[1].wrong_count, kv[0]),
    )[:3]
    if weak:
        parts.append("薄弱考点：" + "、".join(f"{t}（错 {m.wrong_count} 次）" for t, m in weak))
    return _clip("；".join(parts) + "。", SUMMARY_MAX_LEN)


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n] + "…"


# ---- 序列化（档案长存：读端宽容、写端严格）--------------------------------

def to_dict(profile: StudentProfile) -> dict[str, Any]:
    """画像 → 可 JSON 序列化的 dict（含版本号）。"""
    return {
        "version": 1,
        "session_id": profile.session_id,
        "turn_count": profile.turn_count,
        "mastery": {
            topic: {
                "level": m.level,
                "correct_count": m.correct_count,
                "wrong_count": m.wrong_count,
            }
            for topic, m in profile.mastery.items()
        },
        "error_records": [
            {
                "question": r.question,
                "topic": r.topic,
                "error_layer": r.error_layer,
                "solved": r.solved,
                "timestamp": r.timestamp,
            }
            for r in profile.error_records
        ],
        "error_layers": dict(profile.error_layers),
        "last_turn": {
            "level": profile.last_turn.level,
            "topic": profile.last_turn.topic,
            "stuck_point": profile.last_turn.stuck_point,
        },
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def from_dict(data: dict[str, Any]) -> StudentProfile:
    """宽容解析：缺失字段取默认、未知字段忽略、坏嵌套条目跳过。"""
    profile = StudentProfile()
    profile.session_id = _as_str(data.get("session_id"))
    profile.turn_count = max(0, _as_int(data.get("turn_count")))
    mastery = data.get("mastery")
    if isinstance(mastery, dict):
        for topic, raw in mastery.items():
            if not isinstance(topic, str) or not topic or not isinstance(raw, dict):
                continue
            profile.mastery[topic] = TopicMastery(
                level=_as_str(raw.get("level")),
                correct_count=max(0, _as_int(raw.get("correct_count"))),
                wrong_count=max(0, _as_int(raw.get("wrong_count"))),
            )
    records = data.get("error_records")
    if isinstance(records, list):
        for raw in records:
            if not isinstance(raw, dict):
                continue
            profile.error_records.append(
                ErrorRecord(
                    question=_as_str(raw.get("question")),
                    topic=_as_str(raw.get("topic")),
                    error_layer=_as_str(raw.get("error_layer")),
                    solved=bool(raw.get("solved")),
                    timestamp=_as_float(raw.get("timestamp")),
                )
            )
    layers = data.get("error_layers")
    if isinstance(layers, dict):
        for layer, count in layers.items():
            if isinstance(layer, str) and layer:
                value = max(0, _as_int(count))
                if value > 0:  # 零计数条目丢弃（摘要不出现"×0"）
                    profile.error_layers[layer] = value
    last = data.get("last_turn")
    if isinstance(last, dict):
        profile.last_turn = LastTurn(
            level=_as_str(last.get("level")),
            topic=_as_str(last.get("topic")),
            stuck_point=_as_str(last.get("stuck_point")),
        )
    profile.created_at = _as_float(data.get("created_at"))
    profile.updated_at = _as_float(data.get("updated_at"))
    return profile
