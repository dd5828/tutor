"""FastAPI 应用工厂：注册内置能力 + 挂载路由。

能力注册表是全局单例，重复注册同名能力会覆盖（create_app 幂等）。
"""

from __future__ import annotations

from fastapi import FastAPI

from nnnu.api.chat_ws import router
from nnnu.capabilities.chat.capability import ChatCapability
from nnnu.capabilities.echo.capability import EchoCapability
from nnnu.runtime.registry import get_capability_registry


def create_app() -> FastAPI:
    registry = get_capability_registry()
    registry.register(ChatCapability())
    registry.register(EchoCapability())

    app = FastAPI(title="nnnu", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
