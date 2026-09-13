"""学习记忆服务（Phase 3 W7）：学习者画像的结构化存储与确定性摘要。

画像按 session_id 键控（v1 无账号体系，前端页级 UUID 即身份），持久化于
data/memory/（gitignore 覆盖，本地不入库）。记忆是教学的增强信息而非
前置条件：一切读写失败都以"无画像"降级，绝不打断教学主流程。
"""

from __future__ import annotations

from pathlib import Path

# 仓库根目录（src/nnnu/services/memory/__init__.py → 上溯 4 层）
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMORY_DIR = REPO_ROOT / "data" / "memory"
