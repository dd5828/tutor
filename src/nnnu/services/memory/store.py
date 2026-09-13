"""画像档案存取（Phase 3 W7）：路径校验 / 宽容读 / 原子写。

档案布局：data/memory/{session_id}.json。读失败一律返回 None（调用方
按"无画像"降级，记忆故障不打断教学）；写用 tmp + os.replace 原子替换
（同卷 MoveFileEx，写坏不损坏旧档案）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from nnnu.services.memory.profile import StudentProfile, from_dict, to_dict

# 字母数字开头、1-128 字符，允许 . _ -（UUID 兼容）；拒绝路径穿越与空串
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def profile_path(memory_dir: Path, session_id: str) -> Path | None:
    """session_id → 档案路径；非法（空/路径穿越字符/超长）返回 None。"""
    if not _SESSION_ID_RE.fullmatch(session_id):
        return None
    return memory_dir / f"{session_id}.json"


def load_profile(path: Path) -> StudentProfile | None:
    """读档案；不存在/JSON 损坏/结构非法 → None（不抛异常）。"""
    if not path.is_file():
        return None
    try:
        # read_bytes：非法 UTF-8 由 json 统一报 JSONDecodeError（read_text 会抛
        # UnicodeDecodeError 逃出捕获面）
        data = json.loads(path.read_bytes())
    except (json.JSONDecodeError, OSError):
        return None
    return from_dict(data) if isinstance(data, dict) else None


def save_profile(path: Path, profile: StudentProfile) -> None:
    """原子写档案（tmp + os.replace）；目录自动创建；IO 失败抛 OSError。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(to_dict(profile), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)
