"""core/context.py 单测。"""

from __future__ import annotations

from nnnu.core.context import Attachment, TurnContext


def test_defaults() -> None:
    ctx = TurnContext()
    assert ctx.session_id == "" and ctx.user_message == ""
    assert ctx.conversation_history == []
    assert ctx.active_capability is None
    assert ctx.enabled_tools is None  # None = 未指定（默认）
    assert ctx.attachments == []
    assert ctx.language == "zh"  # v1 只做中文
    assert ctx.student_profile == ""
    assert ctx.metadata == {}


def test_enabled_tools_none_vs_empty_distinction() -> None:
    """None（未指定）与 []（显式全禁）是不同语义，必须保留。"""
    assert TurnContext().enabled_tools is None
    assert TurnContext(enabled_tools=[]).enabled_tools == []
    assert TurnContext(enabled_tools=[]).enabled_tools is not None


def test_list_fields_isolated_between_instances() -> None:
    c1 = TurnContext()
    c2 = TurnContext()
    c1.conversation_history.append({"role": "user", "content": "hi"})
    c1.attachments.append(Attachment(type="image"))
    c1.metadata["k"] = "v"
    assert c2.conversation_history == []
    assert c2.attachments == []
    assert c2.metadata == {}


def test_attachment_defaults() -> None:
    att = Attachment(type="pdf")
    assert att.url == "" and att.base64 == ""
    assert att.filename == "" and att.mime_type == ""
    assert att.id == "" and att.extracted_text == ""
