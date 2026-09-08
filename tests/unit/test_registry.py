"""ToolRegistry / CapabilityRegistry 单测。"""

from __future__ import annotations

from typing import Any

import pytest

from nnnu.core import (
    BaseCapability,
    BaseTool,
    CapabilityManifest,
    StreamBus,
    ToolDefinition,
    ToolResult,
    TurnContext,
)
from nnnu.runtime.registry import (
    CapabilityRegistry,
    ToolRegistry,
    get_capability_registry,
    get_tool_registry,
)


class StubTool(BaseTool):
    def __init__(self, name: str, result: str = "ok") -> None:
        self._name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(name=self._name, description=f"{self._name} 测试工具")

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(content=self._result)


class StubCapability(BaseCapability):
    def __init__(self, name: str) -> None:
        self.manifest = CapabilityManifest(
            name=name,
            description=f"{name} 测试能力",
            stages=["阶段一"],
            tools_used=["stub"],
        )

    async def run(self, context: TurnContext, bus: StreamBus) -> None:
        pass


# ---- ToolRegistry ----


def test_register_and_get_and_list() -> None:
    registry = ToolRegistry()
    a, b = StubTool("a"), StubTool("b")
    registry.register(a)
    registry.register(b)

    assert registry.get("a") is a
    assert registry.get("b") is b
    assert registry.list_tools() == ["a", "b"]


def test_get_unknown_returns_none() -> None:
    registry = ToolRegistry()
    assert registry.get("ghost") is None


def test_unregister_removes_tool_and_is_noop_when_absent() -> None:
    registry = ToolRegistry()
    registry.register(StubTool("a"))

    registry.unregister("a")
    assert registry.get("a") is None
    registry.unregister("a")  # 不存在时无操作
    assert registry.list_tools() == []


def test_get_enabled_skips_unknown_and_dedups() -> None:
    registry = ToolRegistry()
    a = StubTool("a")
    registry.register(a)

    enabled = registry.get_enabled(["a", "ghost", "a"])
    assert enabled == [a]


def test_get_definitions_all_and_subset() -> None:
    registry = ToolRegistry()
    registry.register(StubTool("a"))
    registry.register(StubTool("b"))

    all_defs = registry.get_definitions()
    assert [d.name for d in all_defs] == ["a", "b"]

    subset = registry.get_definitions(["b", "ghost"])
    assert [d.name for d in subset] == ["b"]


def test_build_openai_schemas() -> None:
    registry = ToolRegistry()
    registry.register(StubTool("a"))

    schemas = registry.build_openai_schemas()
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "a",
                "description": "a 测试工具",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]


async def test_execute_dispatches_kwargs() -> None:
    registry = ToolRegistry()
    tool = StubTool("a")
    registry.register(tool)

    result = await registry.execute("a", query="你好")

    assert tool.calls == [{"query": "你好"}]
    assert result.content == "ok"


async def test_execute_unknown_raises_keyerror() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="未知工具: ghost"):
        await registry.execute("ghost")


async def test_execute_name_keyword_goes_to_tool_not_registry() -> None:
    """execute 的 name 仅限位置传参，工具自身的 name 参数不被抢占。"""
    registry = ToolRegistry()
    tool = StubTool("a")
    registry.register(tool)

    await registry.execute("a", name="自定义")

    assert tool.calls == [{"name": "自定义"}]


async def test_alias_resolves_with_default_kwargs_and_caller_wins() -> None:
    registry = ToolRegistry(aliases={"快捷名": ("a", {"默认": 1})})
    tool = StubTool("a")
    registry.register(tool)

    await registry.execute("快捷名", 默认=9)
    assert tool.calls == [{"默认": 9}]

    await registry.execute("快捷名")
    assert tool.calls[1] == {"默认": 1}


def test_alias_get_returns_real_tool() -> None:
    registry = ToolRegistry(aliases={"快捷名": ("a", {})})
    tool = StubTool("a")
    registry.register(tool)

    assert registry.get("快捷名") is tool


async def test_alias_to_unknown_tool_execute_raises() -> None:
    registry = ToolRegistry(aliases={"ghost": ("missing", {})})
    with pytest.raises(KeyError, match="未知工具: ghost"):
        await registry.execute("ghost")


# ---- CapabilityRegistry ----


def test_capability_register_and_get() -> None:
    registry = CapabilityRegistry()
    cap = StubCapability("echo")
    registry.register(cap)

    assert registry.get("echo") is cap
    assert registry.get("ghost") is None
    assert registry.list_capabilities() == ["echo"]


def test_capability_register_same_name_overwrites() -> None:
    registry = CapabilityRegistry()
    registry.register(StubCapability("echo"))
    second = StubCapability("echo")
    registry.register(second)

    assert registry.get("echo") is second
    assert registry.list_capabilities() == ["echo"]


def test_get_manifests_shape() -> None:
    registry = CapabilityRegistry()
    registry.register(StubCapability("echo"))

    assert registry.get_manifests() == [
        {
            "name": "echo",
            "description": "echo 测试能力",
            "stages": ["阶段一"],
            "tools_used": ["stub"],
        }
    ]


# ---- 全局单例 ----


def test_singleton_accessors_return_same_instance() -> None:
    tool_registry = get_tool_registry()
    cap_registry = get_capability_registry()

    assert get_tool_registry() is tool_registry
    assert get_capability_registry() is cap_registry
    assert isinstance(tool_registry, ToolRegistry)
    assert isinstance(cap_registry, CapabilityRegistry)
