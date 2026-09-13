"""判答工具（Phase 2 W6）：包装 pedagogy.judge_answer。

v1 判答仍走 LLM 结构化输出（数学符号判题短期无规则解），工具化是为了
把"判答"变成可替换的接缝：未来换成规则/sympy 实现只改本文件，能力层
stage 流程不动。judge_answer 的提示词、temperature、保守降级策略全部
原样继承。

与 rag 工具的错误语义**相反**：judge_answer 的 ChatError 不捕获、照常
抛出——知识库失败可静默降级，判答 LLM 故障必须中断本轮（捕获成
success=False 会让能力层回退直调，同一次故障重复计费）。

``on_usage`` 是能力层→工具的记账回传参数（轮末用量汇总用），**不进
ToolDefinition schema**：schema 只描述 LLM 可调用的参数面。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from nnnu.capabilities.math.pedagogy import judge_answer
from nnnu.core import BaseTool, ToolDefinition, ToolParameter, ToolResult
from nnnu.services.llm import LLMClient


class CheckAnswerTool(BaseTool):
    """判断学生作答：对错 + 错因层 + 下一步建议。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        # 测试注入假 LLMClient；生产用默认客户端（延迟构造，无 key 不报错）
        self._llm = llm if llm is not None else LLMClient()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="check_answer",
            description=(
                "判断学生的作答：返回对错、错因层（概念/公式/方法/计算）与"
                "下一步建议（continue/solved/explain）。引导阶段判答时调用。"
            ),
            parameters=[
                ToolParameter(
                    name="question", type="string", required=True, description="当前题目"
                ),
                ToolParameter(
                    name="last_hint", type="string", required=True, description="上一轮给学生的提示"
                ),
                ToolParameter(
                    name="reply", type="string", required=True, description="学生的作答"
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        question = str(kwargs.get("question", "")).strip()
        reply = str(kwargs.get("reply", "")).strip()
        if not question or not reply:
            return ToolResult(content="判答需要 question 与 reply，不能为空", success=False)
        on_usage = kwargs.get("on_usage")
        judgement = await judge_answer(
            self._llm,
            question=question,
            last_hint=str(kwargs.get("last_hint", "")).strip(),
            reply=reply,
            on_usage=on_usage if callable(on_usage) else None,
        )
        data = asdict(judgement)
        return ToolResult(
            content=json.dumps(data, ensure_ascii=False),
            metadata={"judgement": data},
            success=True,
        )
