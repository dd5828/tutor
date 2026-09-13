"""RAG 知识库服务（Phase 2 W4）：解析 → 分块 → 嵌入 → 索引 → 检索。

v1 约定：
- 语料、索引与模型缓存都在仓库根的 data/ 下（gitignore 覆盖，本地不入库）；
- 嵌入用本地 fastembed 模型（可选依赖 [rag]），缺失时服务优雅降级；
- 向量库用 numpy 内存余弦检索，接口已隔离，后期可换 FAISS。
"""

from __future__ import annotations

from pathlib import Path

# 仓库根目录（src/nnnu/services/rag/__init__.py → 上溯 4 层）
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "rag_corpus"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "rag_index"
# 模型缓存显式指定：Windows 上 fastembed 默认缓存落 %TEMP% 会被系统清理，
# 这里与语料/索引同放 data/ 下。
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / ".fastembed_cache"
