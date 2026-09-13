"""画像存储单测：session_id 校验 / 宽容读 / 原子写（tmp_path 注入）。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nnnu.services.memory.profile import StudentProfile, TurnRecord, apply_turn, to_dict
from nnnu.services.memory.store import load_profile, profile_path, save_profile


def _profile() -> StudentProfile:
    profile = StudentProfile(session_id="s1")
    apply_turn(
        profile,
        TurnRecord(question="题", topic="泰勒公式", error_layer="公式", timestamp=1.0),
    )
    return profile


def test_profile_path_rejects_empty_and_invalid(tmp_path: Any) -> None:
    assert profile_path(tmp_path, "") is None
    assert profile_path(tmp_path, "a/b") is None
    assert profile_path(tmp_path, "..") is None
    assert profile_path(tmp_path, "a" * 200) is None
    assert profile_path(tmp_path, "-abc") is None  # 非字母数字开头


def test_profile_path_accepts_uuid(tmp_path: Any) -> None:
    sid = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    path = profile_path(tmp_path, sid)
    assert path is not None
    assert path.name == f"{sid}.json"


def test_load_missing_returns_none(tmp_path: Any) -> None:
    assert load_profile(tmp_path / "不存在.json") is None


def test_load_corrupt_json_returns_none(tmp_path: Any) -> None:
    path = tmp_path / "s1.json"
    path.write_bytes(b"\x00\xff\xfe\xfd")  # 非法 UTF-8 垃圾字节
    assert load_profile(path) is None


def test_load_non_dict_returns_none(tmp_path: Any) -> None:
    path = tmp_path / "s1.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_profile(path) is None


def test_save_load_round_trip(tmp_path: Any) -> None:
    profile = _profile()
    path = tmp_path / "深层" / "s1.json"  # 目录自动创建
    save_profile(path, profile)
    loaded = load_profile(path)
    assert loaded == profile


def test_save_is_atomic_no_tmp_left(tmp_path: Any) -> None:
    path = tmp_path / "s1.json"
    save_profile(path, _profile())
    assert [p.name for p in tmp_path.iterdir()] == ["s1.json"]


def test_save_failure_raises_oserror(tmp_path: Any) -> None:
    # 父目录路径被一个同名文件占位 → mkdir 抛 FileExistsError（OSError 子类）
    (tmp_path / "s1.json").write_text("我是文件不是目录", encoding="utf-8")
    with pytest.raises(OSError):
        save_profile(tmp_path / "s1.json" / "nested.json", _profile())


def test_saved_json_contains_version(tmp_path: Any) -> None:
    path = tmp_path / "s1.json"
    save_profile(path, _profile())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert to_dict(_profile())["version"] == 1
