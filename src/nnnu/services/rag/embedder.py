"""文本嵌入器（Phase 2 W4）。

Embedder 协议 + FastEmbedEmbedder（本地 ONNX 中文模型，可选依赖）。
设计要点：
- 同步接口：bge-small 单条查询 CPU 推理几十毫秒，索引构建离线跑，无需异步；
- 延迟构造：首次 embed 才 import fastembed 并加载模型（应用启动零成本，
  未安装 [rag] 依赖时只有真正用到才报错）；
- 维度不进协议：由实现与索引数据决定，全链路不硬编码。
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

from nnnu.core import ChatError
from nnnu.services.rag import DEFAULT_CACHE_DIR

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class Embedder(ABC):
    """文本嵌入协议。实现类：FastEmbedEmbedder（生产）、测试用 FakeEmbedder。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """段落嵌入：返回 shape (len(texts), dim) 的 float32 数组；texts 非空。"""
        ...

    def query_embed(self, text: str) -> np.ndarray:
        """查询嵌入：默认退化为 embed([text])[0]。

        bge 族模型查询侧建议带指令前缀，实现类可覆盖本方法用模型自带的
        query_embed。
        """
        return cast(np.ndarray, self.embed([text])[0])


class FastEmbedEmbedder(Embedder):
    """fastembed + BAAI/bge-small-zh-v1.5（512 维中文模型，ONNX CPU 推理）。

    ``cache_dir`` 显式指定（默认 data/.fastembed_cache）：fastembed 默认缓存
    落 %TEMP%，Windows 存储感知清理会损坏缓存；CLI 与 RagTool 共用
    DEFAULT_CACHE_DIR 常量，避免换目录重复下载模型。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get("FASTEMBED_CACHE_PATH") or str(
            DEFAULT_CACHE_DIR
        )
        self._model: Any = None  # 延迟构造：首次 embed 才下载/加载模型

    def _get_model(self) -> Any:
        if self._model is None:
            # importlib 动态导入：fastembed 是可选依赖，mypy strict 不解析
            # 未安装的第三方模块；CI（只装 [dev]）下本方法只在测试中触发。
            try:
                fastembed = importlib.import_module("fastembed")
            except ImportError as exc:
                raise ChatError(
                    "未安装 fastembed：请执行 pip install -e '.[rag]'",
                    retryable=False,
                    error_code="RAG_EMBEDDER_MISSING",
                ) from exc
            try:
                self._model = fastembed.TextEmbedding(
                    model_name=self._model_name, cache_dir=self._cache_dir
                )
            except Exception as exc:
                raise ChatError(
                    f"嵌入模型加载失败（首次使用需联网下载，缓存目录 {self._cache_dir}）: {exc}",
                    retryable=False,
                    error_code="RAG_MODEL_ERROR",
                ) from exc
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts 不能为空")
        try:
            # fastembed 的 embed 返回生成器，逐条 yield 一维 float32 数组
            return np.asarray(list(self._get_model().embed(texts)), dtype=np.float32)
        except ChatError:
            raise
        except Exception as exc:
            raise ChatError(
                f"嵌入失败: {exc}", retryable=False, error_code="RAG_EMBED_ERROR"
            ) from exc

    def query_embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        # 模型支持查询专用方法时用之（bge 查询侧带指令前缀）；不支持则回退
        query_embed = getattr(model, "query_embed", None)
        if query_embed is None:
            return cast(np.ndarray, self.embed([text])[0])
        return cast(np.ndarray, np.asarray(list(query_embed([text])), dtype=np.float32)[0])
