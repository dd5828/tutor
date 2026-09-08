"""core/tool_protocol.py 单测。"""

from __future__ import annotations

from typing import Any

import pytest

from nnnu.core.tool_protocol import BaseTool, ToolDefinition, ToolParameter, ToolResult


def test_parameter_to_schema_basic() -> None:
    p = ToolParameter(name="query", type="string", description="搜索词")
    assert p.to_schema() == {"type": "string", "description": "搜索词"}


def test_parameter_to_schema_enum() -> None:
    p = ToolParameter(name="level", type="string", enum=["易", "中", "难"])
    assert p.to_schema()["enum"] == ["易", "中", "难"]


def test_array_parameter_without_items_falls_back_to_string() -> None:
    """type="array" 且 items=None 时回退 {"type": "string"}（严格供应商 400 坑）。"""
    p = ToolParameter(name="tags", type="array")
    assert p.to_schema()["items"] == {"type": "string"}


def test_array_parameter_with_items_passes_through() -> None:
    items = {"type": "integer"}
    p = ToolParameter(name="ids", type="array", items=items)
    assert p.to_schema()["items"] == items


def test_definition_to_openai_schema() -> None:
    definition = ToolDefinition(
        name="check_answer",
        description="检查学生答案",
        parameters=[
            ToolParameter(name="answer", type="string"),
            ToolParameter(name="expected", type="string"),
            ToolParameter(name="hint", type="boolean", required=False),
        ],
    )
    schema = definition.to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "check_answer"
    assert fn["parameters"]["type"] == "object"
    assert set(fn["parameters"]["properties"]) == {"answer", "expected", "hint"}
    assert fn["parameters"]["required"] == ["answer", "expected"]  # hint 非必填不入列


def test_result_defaults_and_str() -> None:
    r = ToolResult()
    assert r.content == "" and r.sources == [] and r.metadata == {}
    assert r.success is True
    assert str(ToolResult(content="对了")) == "对了"


class EchoTool(BaseTool):
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(name="echo", description="原样返回")

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(content=str(kwargs))


def test_base_tool_deferred_defaults_false() -> None:
    assert EchoTool.deferred is False  # 渐进披露字段保留但 v1 恒 False


def test_base_tool_name_property() -> None:
    assert EchoTool().name == "echo"


def test_base_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract]
