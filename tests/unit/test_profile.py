"""画像纯函数单测：apply_turn / 序列化宽容度 / 确定性摘要。"""

from __future__ import annotations

from nnnu.services.memory.profile import (
    MAX_ERROR_RECORDS,
    StudentProfile,
    TurnRecord,
    apply_turn,
    from_dict,
    profile_summary,
    to_dict,
)


def _turn(
    question: str = "题",
    topic: str = "泰勒公式",
    error_layer: str = "公式",
    solved: bool = False,
    level: str = "常规",
    stuck_point: str = "卡在展开",
    timestamp: float = 100.0,
) -> TurnRecord:
    return TurnRecord(
        question=question,
        topic=topic,
        error_layer=error_layer,
        solved=solved,
        level=level,
        stuck_point=stuck_point,
        timestamp=timestamp,
    )


def test_from_dict_to_dict_round_trip() -> None:
    profile = StudentProfile(session_id="s1")
    apply_turn(profile, _turn())
    apply_turn(profile, _turn(topic="洛必达法则", error_layer="", solved=True))
    restored = from_dict(to_dict(profile))
    assert restored == profile


def test_from_dict_tolerates_missing_fields() -> None:
    profile = from_dict({})
    assert profile == StudentProfile()
    assert profile.turn_count == 0 and profile.mastery == {} and profile.error_records == []


def test_from_dict_ignores_unknown_fields() -> None:
    profile = from_dict({"turn_count": 3, "future_field": {"x": 1}})
    assert profile.turn_count == 3


def test_from_dict_bad_nested_entry_skipped() -> None:
    data = to_dict(StudentProfile(session_id="s1"))
    data["error_records"] = ["不是对象", {"question": "ok", "error_layer": "公式"}]
    profile = from_dict(data)
    assert len(profile.error_records) == 1
    assert profile.error_records[0].question == "ok"


def test_from_dict_non_int_counts_zeroed() -> None:
    profile = from_dict({"turn_count": "abc", "error_layers": {"公式": "很多"}})
    assert profile.turn_count == 0
    assert profile.error_layers == {}


def test_apply_turn_first_turn_sets_created_at() -> None:
    profile = StudentProfile()
    apply_turn(profile, _turn(timestamp=42.0))
    assert profile.created_at == 42.0
    assert profile.updated_at == 42.0
    assert profile.turn_count == 1


def test_apply_turn_updates_mastery() -> None:
    profile = StudentProfile()
    apply_turn(profile, _turn(topic="泰勒公式", solved=False))
    apply_turn(profile, _turn(topic="泰勒公式", solved=True, level="提高"))
    apply_turn(profile, _turn(topic="洛必达法则", solved=True))
    taylor = profile.mastery["泰勒公式"]
    assert taylor.wrong_count == 1 and taylor.correct_count == 1
    assert taylor.level == "提高"  # level 随最近一轮覆盖
    assert profile.mastery["洛必达法则"].correct_count == 1


def test_apply_turn_error_record_and_layer_counter() -> None:
    profile = StudentProfile()
    apply_turn(profile, _turn(error_layer="公式"))
    apply_turn(profile, _turn(error_layer="公式"))
    apply_turn(profile, _turn(error_layer="概念"))
    assert profile.error_layers == {"公式": 2, "概念": 1}
    assert len(profile.error_records) == 3


def test_apply_turn_solved_turn_no_error_record() -> None:
    profile = StudentProfile()
    apply_turn(profile, _turn(error_layer="", solved=True))
    assert profile.error_records == []
    assert profile.mastery["泰勒公式"].correct_count == 1


def test_apply_turn_trims_error_records_to_max() -> None:
    profile = StudentProfile()
    for i in range(MAX_ERROR_RECORDS + 3):
        apply_turn(profile, _turn(question=f"题{i}", timestamp=float(i)))
    assert len(profile.error_records) == MAX_ERROR_RECORDS
    # 保留最近 MAX 条
    assert profile.error_records[0].question == "题3"
    assert profile.error_records[-1].question == f"题{MAX_ERROR_RECORDS + 2}"


def test_profile_summary_empty_profile_returns_empty() -> None:
    assert profile_summary(StudentProfile()) == ""


def test_profile_summary_deterministic_format() -> None:
    profile = StudentProfile()
    apply_turn(profile, _turn(error_layer="公式", solved=False))
    apply_turn(profile, _turn(topic="洛必达法则", error_layer="概念", solved=False))
    apply_turn(profile, _turn(topic="泰勒公式", error_layer="公式", solved=False))
    first = profile_summary(profile)
    second = profile_summary(profile)
    assert first == second  # 确定性：两次逐字相等
    assert "3 轮" in first
    assert "卡在" in first and "泰勒公式" in first
    assert "公式×2" in first and "概念×1" in first  # 高频错因排前
    assert "薄弱考点" in first
    assert len(first) <= 200
