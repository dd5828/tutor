"""FastAPI 应用工厂：注册内置能力 + 挂载路由与静态前端。

能力注册表是全局单例，重复注册同名能力会覆盖（create_app 幂等）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nnnu.api.chat_ws import router
from nnnu.capabilities.chat.capability import ChatCapability
from nnnu.capabilities.echo.capability import EchoCapability
from nnnu.capabilities.quiz.capability import QuizCapability
from nnnu.runtime.registry import get_capability_registry

# 仓库根目录下的 web/（src/nnnu/api/main.py → 上溯 3 层）
WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app() -> FastAPI:
    registry = get_capability_registry()
    registry.register(ChatCapability())
    registry.register(EchoCapability())
    registry.register(QuizCapability())

    app = FastAPI(title="nnnu", version="0.1.0")
    app.include_router(router)
    # 最小聊天前端挂在根路径；路由先注册，/chat 优先匹配
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
