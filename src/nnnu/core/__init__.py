"""协议层（第 5 层）：最先写、最先冻结。

- events.py              StreamEvent + StreamEventType
- bus.py                 StreamBus 总线
- context.py             TurnContext 上下文
- tool_protocol.py       BaseTool + ToolDefinition + ToolResult
- capability_protocol.py BaseCapability + CapabilityManifest
- errors.py              统一异常 ChatError

冻结后改动走评审（路线图 Phase 0 Day 3）。
"""

from nnnu.core.bus import StreamBus
from nnnu.core.capability_protocol import BaseCapability, CapabilityManifest
from nnnu.core.context import Attachment, TurnContext
from nnnu.core.errors import ChatError
from nnnu.core.events import StreamEvent, StreamEventType
from nnnu.core.tool_protocol import BaseTool, ToolDefinition, ToolParameter, ToolResult

__all__ = [
    "Attachment",
    "BaseCapability",
    "BaseTool",
    "CapabilityManifest",
    "ChatError",
    "StreamBus",
    "StreamEvent",
    "StreamEventType",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "TurnContext",
]
