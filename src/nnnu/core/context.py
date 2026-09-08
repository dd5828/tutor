"""TurnContext 上下文信封（v1 精简版）。

一轮对话中贯穿 orchestrator → capability → tool 的唯一数据对象。
v1 共 9 字段 + 1 教育特色：session_id / user_message / conversation_history /
active_capability / enabled_tools / attachments / language / student_profile /
metadata。其余信息走 metadata 兜底或能力自备（如知识库名由 rag 工具参数传递）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attachment:
    """用户消息携带的文件或图片。"""

    type: str  # "image" | "file" | "pdf"
    url: str = ""
    base64: str = ""
    filename: str = ""
    mime_type: str = ""
    # 附件级稳定标识；原始字节在附件存储中按此目录段存放。
    id: str = ""
    # 二进制文档（PDF/DOCX 等）的纯文本渲染，供前端预览"模型看到了什么"。
    extracted_text: str = ""


@dataclass
class TurnContext:
    """能力/工具处理一轮对话所需的全部上下文。

    Attributes:
        session_id: 持久会话标识。
        user_message: 当前用户输入。
        conversation_history: 历史消息（OpenAI 格式）。
        active_capability: 用户选定的能力名；None = 走默认 chat。
        enabled_tools: 用户开启的工具名。**None 表示"未指定"，[] 表示
            "显式禁用全部可选工具"** —— None 与 [] 语义不同，调用方必须区分。
        attachments: 随消息携带的文件/图片。
        language: 界面/回复语言（v1 只做中文，默认 "zh"）。
        student_profile: ⭐ 教育特色：学习者画像注入（Phase 3 使用）。
        metadata: 兜底字段，能力间传临时数据时不用动协议。
    """

    session_id: str = ""
    user_message: str = ""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    active_capability: str | None = None
    enabled_tools: list[str] | None = None
    attachments: list[Attachment] = field(default_factory=list)
    language: str = "zh"
    student_profile: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
