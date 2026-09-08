"""LLMClient 单测（注入假 AsyncOpenAI，不触网）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from nnnu.core import ChatError
from nnnu.services.llm import LLMClient
from nnnu.services.usage import UsageInfo


def _chunk(text: str, usage: Any | None = None) -> Any:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text else []
    return SimpleNamespace(choices=choices, usage=usage)


class FakeStream:
    """模拟 SDK 的流对象：可异步迭代，error 非空时在产出完 chunk 后抛出。

    usage 非空时，在所有正文 chunk 之后追加一个「空 choices + usage」尾块
    （include_usage 开启时供应商的真实行为）。
    """

    def __init__(
        self,
        chunks: list[str],
        error: Exception | None = None,
        usage: Any | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._usage = usage

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        for text in self._chunks:
            yield _chunk(text)
        if self._usage is not None:
            yield _chunk("", usage=self._usage)
        if self._error is not None:
            raise self._error


class FakeCompletions:
    """create 是普通协程（与 SDK 一致），返回可异步迭代的流对象。"""

    def __init__(
        self,
        chunks: list[str],
        error: Exception | None = None,
        usage: Any | None = None,
    ) -> None:
        self._stream = FakeStream(chunks, error, usage)
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._stream


def _client(completions: FakeCompletions, model: str = "m") -> tuple[LLMClient, FakeCompletions]:
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(cast(AsyncOpenAI, fake), model=model), completions


async def _collect(agen: AsyncIterator[str]) -> list[str]:
    return [text async for text in agen]


async def test_chat_stream_yields_text_chunks() -> None:
    client, _ = _client(FakeCompletions(["你", "好"]))

    texts = await _collect(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert texts == ["你", "好"]


async def test_chat_stream_passes_messages_model_and_flags() -> None:
    client, completions = _client(FakeCompletions(["x"]))
    messages = [{"role": "user", "content": "hi"}]

    await _collect(client.chat_stream(messages, temperature=0.1))

    call = completions.create_calls[0]
    assert call["messages"] == messages
    assert call["model"] == "m"
    assert call["stream"] is True
    assert call["temperature"] == 0.1


async def test_chat_stream_skips_empty_delta() -> None:
    """SDK 偶发返回空 delta（content 为空串），不应产生空片段。"""
    client, _ = _client(FakeCompletions(["", "你"]))

    texts = await _collect(client.chat_stream([]))

    assert texts == ["你"]


async def test_chat_stream_error_wraps_in_chat_error_with_partial() -> None:
    client, _ = _client(FakeCompletions(["部分"], error=ConnectionError("断网")))

    with pytest.raises(ChatError) as excinfo:
        await _collect(client.chat_stream([]))

    exc = excinfo.value
    assert exc.retryable is True
    assert exc.partial_response == "部分"
    assert exc.error_code == "LLM_ERROR"
    assert "断网" in str(exc)


async def test_chat_stream_error_without_partial_has_none() -> None:
    client, _ = _client(FakeCompletions([], error=ConnectionError("断网")))

    with pytest.raises(ChatError) as excinfo:
        await _collect(client.chat_stream([]))

    assert excinfo.value.partial_response is None


async def test_chat_error_passthrough_keeps_original() -> None:
    original = ChatError("上游包装过的错误", retryable=False, error_code="E1")
    client, _ = _client(FakeCompletions([], error=original))

    with pytest.raises(ChatError) as excinfo:
        await _collect(client.chat_stream([]))

    assert excinfo.value is original


def _usage_obj(prompt: int = 10, completion: int = 5, total: int = 15) -> Any:
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


async def test_chat_stream_requests_include_usage() -> None:
    client, completions = _client(FakeCompletions(["x"]))

    await _collect(client.chat_stream([]))

    assert completions.create_calls[0]["stream_options"] == {"include_usage": True}


async def test_chat_stream_fires_on_usage_once_with_parsed_tokens() -> None:
    client, _ = _client(FakeCompletions(["你"], usage=_usage_obj()))
    received: list[UsageInfo] = []

    texts = await _collect(client.chat_stream([], on_usage=received.append))

    assert texts == ["你"]  # usage 尾块不产生空正文
    assert len(received) == 1
    assert received[0].prompt_tokens == 10
    assert received[0].completion_tokens == 5
    assert received[0].total_tokens == 15


async def test_chat_stream_usage_only_chunk_without_text_does_not_crash() -> None:
    """流只含 usage 尾块（无任何正文）时正常结束、不 yield、回调触发。"""
    client, _ = _client(FakeCompletions([], usage=_usage_obj()))
    received: list[UsageInfo] = []

    texts = await _collect(client.chat_stream([], on_usage=received.append))

    assert texts == []
    assert len(received) == 1
