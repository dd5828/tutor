"""统一异常。

orchestrator 捕获异常后不 import 本模块，只用
getattr(exc, "error_code"/"retryable"/"partial_response", None) 反射属性
塞进 error 事件 metadata。因此这些属性名是跨模块约定，改名即破坏反射。
"""

from __future__ import annotations


class ChatError(Exception):
    """能力/工具链任何环节抛出的统一异常。"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        partial_response: str | None = None,
        error_code: str = "",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        # 出错前已流式输出给用户的部分正文（前端可保留显示）。
        self.partial_response = partial_response
        self.error_code = error_code
