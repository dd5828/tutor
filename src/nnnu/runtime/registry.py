"""运行时注册表（v1 精简版）。

ToolRegistry 与 CapabilityRegistry 合并于此：注册 / 查询 / 列清单 /
OpenAI schema 组装 / 执行分发。

v1 未做：内置类路径自动加载（Day 5 起由调用方显式 register）、第三方插件
发现、提示词组装（get_prompt_hints / build_prompt_text）、内置工具别名表
（别名解析机制保留，别名映射由调用方注入）。
"""

from __future__ import annotations

from typing import Any

from nnnu.core import BaseCapability, BaseTool, ToolDefinition, ToolResult


class ToolRegistry:
    """所有工具（内置或插件）的注册表。"""

    def __init__(self, aliases: dict[str, tuple[str, dict[str, Any]]] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        # 别名 → (真实工具名, 默认参数)；调用方 kwargs 覆盖默认参数。
        self._aliases: dict[str, tuple[str, dict[str, Any]]] = dict(aliases or {})

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """移除工具（不存在时无操作）。"""
        self._tools.pop(name, None)

    def _resolve_request(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if name in self._tools:
            return name, dict(kwargs or {})
        resolved_name, default_kwargs = self._aliases.get(name, (name, {}))
        return resolved_name, {**default_kwargs, **(kwargs or {})}

    def get(self, name: str) -> BaseTool | None:
        resolved_name, _ = self._resolve_request(name)
        return self._tools.get(resolved_name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_enabled(self, names: list[str]) -> list[BaseTool]:
        """返回给定名称对应的工具实例（跳过未知名称，按真实名去重）。"""
        enabled: list[BaseTool] = []
        seen: set[str] = set()
        for name in names:
            tool = self.get(name)
            if tool is None or tool.name in seen:
                continue
            enabled.append(tool)
            seen.add(tool.name)
        return enabled

    def get_definitions(self, names: list[str] | None = None) -> list[ToolDefinition]:
        """返回 *names* 的工具定义（None = 全部）。"""
        tools = self._tools.values() if names is None else self.get_enabled(names)
        return [t.get_definition() for t in tools]

    def build_openai_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """组装 OpenAI function-calling 工具 schema。"""
        return [d.to_openai_schema() for d in self.get_definitions(names)]

    async def execute(self, name: str, /, **kwargs: Any) -> ToolResult:
        """解析别名后执行工具。

        ``name`` 仅限位置传参，避免与工具自身恰好叫 ``name`` 的参数冲突。
        """
        resolved_name, resolved_kwargs = self._resolve_request(name, kwargs)
        tool = self._tools.get(resolved_name)
        if tool is None:
            raise KeyError(f"未知工具: {name}")
        return await tool.execute(**resolved_kwargs)


class CapabilityRegistry:
    """所有能力（深模式）的注册表。"""

    def __init__(self) -> None:
        self._capabilities: dict[str, BaseCapability] = {}

    def register(self, capability: BaseCapability) -> None:
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> BaseCapability | None:
        return self._capabilities.get(name)

    def list_capabilities(self) -> list[str]:
        return list(self._capabilities.keys())

    def get_manifests(self) -> list[dict[str, Any]]:
        """所有能力的 manifest 摘要（REST 列表接口用）。"""
        return [
            {
                "name": c.name,
                "description": c.manifest.description,
                "stages": c.manifest.stages,
                "tools_used": c.manifest.tools_used,
            }
            for c in self._capabilities.values()
        ]


_default_tool_registry: ToolRegistry | None = None
_default_capability_registry: CapabilityRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """返回全局工具注册表（首次调用创建）。"""
    global _default_tool_registry
    if _default_tool_registry is None:
        _default_tool_registry = ToolRegistry()
    return _default_tool_registry


def get_capability_registry() -> CapabilityRegistry:
    """返回全局能力注册表（首次调用创建）。"""
    global _default_capability_registry
    if _default_capability_registry is None:
        _default_capability_registry = CapabilityRegistry()
    return _default_capability_registry
