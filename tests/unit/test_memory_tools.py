"""记忆工具单测：schema / 读写三态 / 词表钳制 / 永不抛异常（tmp_path 注入）。"""

from __future__ import annotations

from typing import Any

from nnnu.tools.memory_tools import ReadMemoryTool, WriteMemoryTool

SID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def test_read_definition_schema() -> None:
    definition = ReadMemoryTool().get_definition()
    assert definition.name == "read_memory"
    assert definition.to_openai_schema()["function"]["parameters"]["required"] == ["session_id"]


def test_write_definition_schema() -> None:
    definition = WriteMemoryTool().get_definition()
    assert definition.name == "write_memory"
    params = {p.name: p for p in definition.parameters}
    assert params["question"].required and params["session_id"].required
    assert not params["topic"].required and not params["level"].required
    assert not params["error_layer"].required and not params["solved"].required
    assert params["level"].enum == ["基础", "常规", "提高"]
    assert params["error_layer"].enum == ["概念", "公式", "方法", "计算"]


async def test_read_no_file_returns_empty_success(tmp_path: Any) -> None:
    result = await ReadMemoryTool(memory_dir=tmp_path).execute(session_id=SID)
    assert result.success
    assert "暂无历史" in result.content
    assert result.metadata["exists"] is False
    assert result.metadata["summary"] == ""


async def test_read_corrupt_returns_success_with_corrupt_flag(tmp_path: Any) -> None:
    (tmp_path / f"{SID}.json").write_bytes(b"\x00\xff\xfe\xfd")  # 非法 UTF-8
    result = await ReadMemoryTool(memory_dir=tmp_path).execute(session_id=SID)
    assert result.success  # 损坏也不打断教学
    assert result.metadata["corrupt"] is True
    assert "损坏" in result.content


async def test_read_valid_returns_summary_and_metadata(tmp_path: Any) -> None:
    writer = WriteMemoryTool(memory_dir=tmp_path)
    await writer.execute(
        session_id=SID, question="题", topic="泰勒公式",
        error_layer="公式", level="常规", stuck_point="卡在展开",
    )
    result = await ReadMemoryTool(memory_dir=tmp_path).execute(session_id=SID)
    assert result.success
    assert result.metadata["exists"] is True
    assert "1 轮" in result.metadata["summary"]
    assert result.metadata["last_topic"] == "泰勒公式"
    assert result.metadata["last_stuck"] == "卡在展开"
    assert result.metadata["turn_count"] == 1
    assert isinstance(result.metadata["profile"], dict)


async def test_read_invalid_session_fails(tmp_path: Any) -> None:
    result = await ReadMemoryTool(memory_dir=tmp_path).execute(session_id="../etc")
    assert not result.success


async def test_write_first_turn_creates_file(tmp_path: Any) -> None:
    result = await WriteMemoryTool(memory_dir=tmp_path).execute(
        session_id=SID, question="题"
    )
    assert result.success
    assert result.metadata["turn_count"] == 1
    assert (tmp_path / f"{SID}.json").is_file()


async def test_write_accumulates_turns(tmp_path: Any) -> None:
    writer = WriteMemoryTool(memory_dir=tmp_path)
    await writer.execute(session_id=SID, question="题一")
    result = await writer.execute(session_id=SID, question="题二", error_layer="概念")
    assert result.metadata["turn_count"] == 2
    assert "2 轮" in result.metadata["summary"]


async def test_write_empty_question_fails(tmp_path: Any) -> None:
    result = await WriteMemoryTool(memory_dir=tmp_path).execute(session_id=SID, question="  ")
    assert not result.success


async def test_write_invalid_session_id_fails(tmp_path: Any) -> None:
    result = await WriteMemoryTool(memory_dir=tmp_path).execute(
        session_id="..", question="题"
    )
    assert not result.success
    assert not list(tmp_path.iterdir())  # 无文件落盘


async def test_write_clamps_invalid_enums(tmp_path: Any) -> None:
    writer = WriteMemoryTool(memory_dir=tmp_path)
    result = await writer.execute(
        session_id=SID, question="题", level="超难", error_layer="粗心"
    )
    assert result.success
    # 档位钳制为常规、非法错因不记录（summary 无易错点段）
    assert "易错点" not in result.metadata["summary"]


async def test_write_never_raises_on_io_error(tmp_path: Any) -> None:
    # 真实 IO 失败：memory_dir 是文件 → 档案父目录 mkdir 抛 FileExistsError
    blocker = tmp_path / "blocked"
    blocker.write_text("占位", encoding="utf-8")
    result = await WriteMemoryTool(memory_dir=blocker).execute(session_id=SID, question="题")
    assert not result.success  # 不抛异常
    assert "失败" in result.content
