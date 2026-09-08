"""LLM 客户端（v1 只接 DeepSeek，OpenAI 兼容协议）。

全项目只有这里调 LLM：换供应商/加缓存/加重试只改这一处（工程质量规范 §7）。
配置走环境变量：NNNU_LLM_API_KEY / NNNU_LLM_BASE_URL / NNNU_LLM_MODEL。

v1 未做：多供应商网关、自动重试、缓存、token 用量计费（Day 6 起加 usage）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from openai import AsyncOpenAI

from nnnu.core import ChatError
from nnnu.services.usage import UsageInfo

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class LLMClient:
    """OpenAI 兼容协议的流式聊天客户端（默认接入 DeepSeek）。

    测试可注入 ``client``（假 AsyncOpenAI），生产用环境变量构造。
    """

    def __init__(self, client: AsyncOpenAI | None = None, *, model: str | None = None) -> None:
        # 延迟构造 SDK client：模块 import / 无 key 场景不触发 SDK 校验
        # （openai 3.x 在构造时即校验凭据，import 时构造会让应用起不来）。
        self._client = client
        self._model = model or os.environ.get("NNNU_LLM_MODEL") or DEFAULT_MODEL

    def _get_client(self) -> AsyncOpenAI:
        """首次调用时构造 SDK client；未配置密钥给友好错误而非 SDK 异常。"""
        if self._client is None:
            api_key = os.environ.get("NNNU_LLM_API_KEY") or ""
            if not api_key:
                raise ChatError(
                    "未配置 LLM 密钥：请设置环境变量 NNNU_LLM_API_KEY",
                    retryable=False,
                    error_code="LLM_NO_KEY",
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.environ.get("NNNU_LLM_BASE_URL") or DEFAULT_BASE_URL,
            )
        return self._client

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        on_usage: Callable[[UsageInfo], None] | None = None,
    ) -> AsyncIterator[str]:
        """流式调用 LLM，逐段 yield 文本。

        ``on_usage``：流结束前收到用量块时回调一次（include_usage 开启）。
        调用失败抛 ``ChatError(retryable=True)``，``partial_response`` 为
        出错前已产出的文本（orchestrator 反射进 error 事件，前端可保留显示）。
        """
        partial: list[str] = []
        try:
            stream = await self._get_client().chat.completions.create(
                model=model or self._model,
                messages=cast(Any, messages),
                stream=True,
                temperature=temperature,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                # 用量尾块：choices 为空、usage 有值（供应商均如此发）
                usage = getattr(chunk, "usage", None)
                if usage is not None and on_usage is not None:
                    on_usage(UsageInfo.from_openai(usage))
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                text = choices[0].delta.content or ""
                if text:
                    partial.append(text)
                    yield text
        except ChatError:
            raise
        except Exception as exc:
            joined = "".join(partial)
            raise ChatError(
                f"LLM 调用失败: {exc}",
                retryable=True,
                partial_response=joined or None,
                error_code="LLM_ERROR",
            ) from exc
