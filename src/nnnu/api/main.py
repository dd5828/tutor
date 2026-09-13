"""FastAPI 应用工厂：注册内置工具与能力 + 挂载路由与静态前端。

注册表是全局单例，重复注册同名覆盖（create_app 幂等）。工具先于能力注册：
能力默认取全局工具注册表（如 math 的 rag 检索）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from nnnu.api.chat_ws import router
from nnnu.capabilities.chat.capability import ChatCapability
from nnnu.capabilities.echo.capability import EchoCapability
from nnnu.capabilities.math.capability import MathTutorCapability
from nnnu.capabilities.quiz.capability import QuizCapability
from nnnu.runtime.registry import get_capability_registry, get_tool_registry
from nnnu.tools.bank.question_bank import QuestionBankTool
from nnnu.tools.check_answer import CheckAnswerTool
from nnnu.tools.rag_tool import RagTool

# 仓库根目录下的 web/（src/nnnu/api/main.py → 上溯 3 层）
WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app() -> FastAPI:
    # 先注册工具再注册能力：能力默认取全局工具注册表
    get_tool_registry().register(RagTool())
    get_tool_registry().register(QuestionBankTool())
    get_tool_registry().register(CheckAnswerTool())
    registry = get_capability_registry()
    registry.register(ChatCapability())
    registry.register(EchoCapability())
    registry.register(QuizCapability())
    registry.register(MathTutorCapability())

    app = FastAPI(title="nnnu", version="0.1.0")
    app.include_router(router)
    # 最小聊天前端挂在根路径；路由先注册，/chat 优先匹配
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
