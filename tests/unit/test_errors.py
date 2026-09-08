"""core/errors.py 单测。"""

from __future__ import annotations

from nnnu.core.errors import ChatError


def test_defaults() -> None:
    err = ChatError("LLM 超时")
    assert str(err) == "LLM 超时"
    assert err.retryable is False
    assert err.partial_response is None
    assert err.error_code == ""


def test_full_attributes() -> None:
    err = ChatError(
        "网络错误", retryable=True, partial_response="已输出: 第一段", error_code="timeout"
    )
    assert err.retryable is True
    assert err.partial_response == "已输出: 第一段"
    assert err.error_code == "timeout"


def test_attributes_reflectable_by_orchestrator() -> None:
    """orchestrator 不 import 本模块，只反射属性。"""
    err = ChatError("boom", retryable=True)
    assert getattr(err, "error_code", None) == ""
    assert getattr(err, "retryable", None) is True
    assert getattr(err, "partial_response", None) is None
    # 普通异常反射不到这些属性 → None（orchestrator 的兜底路径）
    assert getattr(ValueError("x"), "retryable", None) is None
