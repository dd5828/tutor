"""题库 loader 与 QuestionBankTool 单测（注入假库/临时文件，不触网）。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nnnu.capabilities.math.topics import TOPICS
from nnnu.core import ChatError
from nnnu.tools.bank.question_bank import (
    DEFAULT_BANK_PATH,
    BankItem,
    QuestionBank,
    QuestionBankTool,
    load_question_bank,
)


def _write_bank(tmp_path: Any, items: list[dict[str, Any]]) -> Any:
    path = tmp_path / "bank.json"
    path.write_text(
        json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8"
    )
    return path


def _item(topic: str = "泰勒公式", level: str = "常规", iid: str = "qb-x-01") -> dict[str, Any]:
    return {
        "id": iid,
        "topic": topic,
        "level": level,
        "question": f"题目{iid}",
        "answer": f"答案{iid}",
    }


async def test_load_bank_parses_valid_file(tmp_path: Any) -> None:
    path = _write_bank(tmp_path, [_item()])
    bank = load_question_bank(path)
    assert len(bank.items) == 1
    item = bank.items[0]
    assert item.id == "qb-x-01" and item.topic == "泰勒公式" and item.level == "常规"
    assert item.question == "题目qb-x-01" and item.answer == "答案qb-x-01"


def test_load_bank_rejects_missing_field(tmp_path: Any) -> None:
    raw = _item()
    raw.pop("answer")
    path = _write_bank(tmp_path, [raw])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert exc_info.value.error_code == "BANK_DATA_ERROR"
    assert "answer" in str(exc_info.value)


def test_load_bank_rejects_unknown_topic(tmp_path: Any) -> None:
    path = _write_bank(tmp_path, [_item(topic="奥数")])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "考点词表" in str(exc_info.value)


def test_load_bank_rejects_unknown_level(tmp_path: Any) -> None:
    path = _write_bank(tmp_path, [_item(level="困难")])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "level" in str(exc_info.value)


def test_load_bank_rejects_duplicate_id(tmp_path: Any) -> None:
    path = _write_bank(tmp_path, [_item(iid="dup"), _item(iid="dup")])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "重复" in str(exc_info.value)


def test_load_bank_rejects_block_latex(tmp_path: Any) -> None:
    raw = _item()
    raw["question"] = "题目 \\(块级公式\\) 禁用"
    path = _write_bank(tmp_path, [raw])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "块级" in str(exc_info.value)


def test_load_bank_missing_file_raises(tmp_path: Any) -> None:
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(tmp_path / "不存在.json")
    assert exc_info.value.error_code == "BANK_DATA_ERROR"


def test_load_bank_invalid_json_raises(tmp_path: Any) -> None:
    path = tmp_path / "bank.json"
    path.write_text("{ 不是 JSON", encoding="utf-8")
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert exc_info.value.error_code == "BANK_DATA_ERROR"


def test_load_bank_empty_items_raises(tmp_path: Any) -> None:
    path = _write_bank(tmp_path, [])
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "不能为空" in str(exc_info.value)


def test_load_bank_non_list_items_raises(tmp_path: Any) -> None:
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"items": "不是列表"}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ChatError) as exc_info:
        load_question_bank(path)
    assert "结构错误" in str(exc_info.value)


def test_default_bank_file_valid() -> None:
    """真实 question_bank.json 数据回归：词表对齐、id 唯一、每考点至少 1 题。"""
    bank = load_question_bank(DEFAULT_BANK_PATH)
    assert {item.topic for item in bank.items} == set(TOPICS)
    assert len({item.id for item in bank.items}) == len(bank.items)
    assert all(item.level in ("基础", "常规", "提高") for item in bank.items)
    assert all(item.question.strip() and item.answer.strip() for item in bank.items)
    for topic in TOPICS:
        assert bank.get(topic, "基础") or bank.get(topic, "常规") or bank.get(topic, "提高")


def test_bank_get_filters_by_topic_and_level() -> None:
    bank = QuestionBank(
        items=(
            BankItem(id="a", topic="泰勒公式", level="基础", question="q1", answer="a1"),
            BankItem(id="b", topic="泰勒公式", level="常规", question="q2", answer="a2"),
            BankItem(id="c", topic="洛必达法则", level="基础", question="q3", answer="a3"),
        )
    )
    assert [i.id for i in bank.get("泰勒公式", "基础")] == ["a"]
    assert bank.get("泰勒公式", "提高") == []


def test_get_definition_schema() -> None:
    definition = QuestionBankTool().get_definition()
    assert definition.name == "question_bank"
    params = {p.name: p for p in definition.parameters}
    assert params["topic"].required and params["topic"].enum == list(TOPICS)
    assert not params["level"].required and params["level"].enum == ["基础", "常规", "提高"]
    assert definition.to_openai_schema()["function"]["parameters"]["required"] == ["topic"]


async def test_execute_returns_question_and_metadata() -> None:
    bank = QuestionBank(
        items=(
            BankItem(id="qb-t-01", topic="泰勒公式", level="常规", question="题目", answer="答案"),
        )
    )
    tool = QuestionBankTool(bank=bank)
    result = await tool.execute(topic="泰勒公式", level="常规")
    assert result.success
    assert result.content == "题目"
    assert result.metadata["item"] == {
        "id": "qb-t-01",
        "topic": "泰勒公式",
        "level": "常规",
        "question": "题目",
        "answer": "答案",
    }
    assert result.metadata["total"] == 1


async def test_execute_empty_topic_fails() -> None:
    result = await QuestionBankTool(bank=QuestionBank(items=())).execute(topic="  ")
    assert not result.success
    assert "不能为空" in result.content


async def test_execute_unknown_topic_fails_with_vocab() -> None:
    result = await QuestionBankTool(bank=QuestionBank(items=())).execute(topic="奥数")
    assert not result.success
    assert "未知考点" in result.content
    assert "泰勒公式" in result.content  # 词表出现在指引里


async def test_execute_no_item_for_level_guides() -> None:
    bank = QuestionBank(
        items=(BankItem(id="qb-t-01", topic="泰勒公式", level="基础", question="q", answer="a"),)
    )
    result = await QuestionBankTool(bank=bank).execute(topic="泰勒公式", level="提高")
    assert result.success  # 合法考点库存缺口：success=True
    assert result.metadata["item"] is None
    assert "暂未收录" in result.content


async def test_execute_rotation_cycles_items() -> None:
    bank = QuestionBank(
        items=(
            BankItem(id="a", topic="泰勒公式", level="基础", question="题A", answer="答A"),
            BankItem(id="b", topic="泰勒公式", level="基础", question="题B", answer="答B"),
        )
    )
    tool = QuestionBankTool(bank=bank)
    first = await tool.execute(topic="泰勒公式", level="基础")
    second = await tool.execute(topic="泰勒公式", level="基础")
    third = await tool.execute(topic="泰勒公式", level="基础")
    assert [first, second, third][0].metadata["item_id"] == "a"
    assert [first, second, third][1].metadata["item_id"] == "b"
    assert [first, second, third][2].metadata["item_id"] == "a"  # 轮转回起点


async def test_execute_missing_bank_file_degrades(tmp_path: Any) -> None:
    tool = QuestionBankTool(bank_path=tmp_path / "不存在.json")
    result = await tool.execute(topic="泰勒公式")
    assert not result.success
    assert "题库不可用" in result.content


async def test_execute_default_level_is_regular() -> None:
    bank = QuestionBank(
        items=(
            BankItem(id="a", topic="泰勒公式", level="基础", question="q1", answer="a1"),
            BankItem(id="b", topic="泰勒公式", level="常规", question="q2", answer="a2"),
        )
    )
    result = await QuestionBankTool(bank=bank).execute(topic="泰勒公式")
    assert result.metadata["item_id"] == "b"  # 缺省取常规档
