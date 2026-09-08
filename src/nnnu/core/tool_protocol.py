"""工具协议（v1 精简版）。

所有工具——内置或插件——实现 BaseTool。
v1 未做：raw_parameters 透传路径（MCP 适配器用）、ToolResult 的
terminate_turn / pause_for_user（反问走 StreamBus.wait_for_input）、
ToolAlias / ToolPromptHints / ToolEventSink / ToolLookup / provider_identity、
ToolParameter.default（参数默认值，to_schema 不使用）、
BaseTool.get_prompt_hints（提示词动态组装）。
BaseTool.deferred 字段保留但 v1 恒为 False（渐进披露，写死避免协议升级改签名）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """工具函数调用 schema 中的一个参数。

    Attributes:
        items: type="array" 参数的内层 JSON Schema。**严格供应商
            （Gemini、Anthropic）要求必填**，OpenAI 缺了不报错——漏写直接
            400。type="array" 且 items=None 时回退 {"type": "string"}，
            只声明 ToolParameter(type="array") 也能产出合法 schema。
    """

    name: str
    type: str  # "string" | "integer" | "boolean" | "number" | "array" | "object"
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    items: dict[str, Any] | None = None

    def to_schema(self) -> dict[str, Any]:
        """转为 JSON Schema property dict。"""
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.type == "array":
            schema["items"] = self.items if self.items is not None else {"type": "string"}
        return schema


@dataclass
class ToolDefinition:
    """描述给 LLM 的工具元数据（OpenAI function-calling 格式）。"""

    name: str
    description: str  # 写给 LLM 看的，写不好模型就不调用
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """组装 OpenAI function tool schema（properties + required）。"""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = p.to_schema()
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    """工具执行的标准返回值。

    Attributes:
        content: 回给 LLM 的 role=tool 消息正文。
        sources: 引用溯源（经 stream.sources 呈现）。
        metadata: 自由载荷（结构化 UI 提示等）。
        success: False 标记显式失败路径；LLM 仍可读 content（常为错误信息）。
    """

    content: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def __str__(self) -> str:
        return self.content


class BaseTool(ABC):
    """所有工具的抽象基类。

    ``deferred`` 为渐进披露标记，v1 恒为 False（字段保留，不启用）。
    """

    deferred: bool = False

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """返回工具元数据与参数 schema。"""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """以给定关键字参数执行工具。"""
        ...

    @property
    def name(self) -> str:
        return self.get_definition().name
