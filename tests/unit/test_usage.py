"""UsageInfo / UsageTracker 单测。"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

from nnnu.services.usage import UsageInfo, UsageTracker, get_usage_tracker


def test_from_openai_parses_tokens() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    info = UsageInfo.from_openai(usage)

    assert info == UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_from_openai_missing_fields_default_zero() -> None:
    info = UsageInfo.from_openai(SimpleNamespace())

    assert info == UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def test_cost_default_prices() -> None:
    # 默认价：输入 2 元 / 输出 8 元 每百万 token
    info = UsageInfo(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert info.cost() == 10.0


def test_cost_env_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("NNNU_LLM_PRICE_IN", "4")
    monkeypatch.setenv("NNNU_LLM_PRICE_OUT", "10")
    info = UsageInfo(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert info.cost() == 14.0


def test_cost_env_invalid_falls_back_to_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("NNNU_LLM_PRICE_IN", "不是数字")
    info = UsageInfo(prompt_tokens=1_000_000, completion_tokens=0)

    assert info.cost() == 2.0


def test_tracker_records_json_log(caplog: Any) -> None:
    tracker = UsageTracker()
    with caplog.at_level(logging.INFO, logger="nnnu.usage"):
        tracker.record(
            session_id="s1",
            turn_id="t1",
            capability="chat",
            usage=UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            duration=1.234,
        )

    record = json.loads(caplog.records[0].message)
    assert record == {
        "event": "usage",
        "session_id": "s1",
        "turn_id": "t1",
        "capability": "chat",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": 0.0006,
        "duration": 1.234,
    }


def test_singleton_accessor() -> None:
    tracker = get_usage_tracker()

    assert get_usage_tracker() is tracker
    assert isinstance(tracker, UsageTracker)
