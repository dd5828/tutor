"""考研数学题库：校验式 loader + 按考点/档位取题工具（Phase 2 W6）。

数据约定（question_bank.json）：
- 顶层 {"items": [...]}，每条必填 id / topic / level / question / answer；
- topic ∈ 考点词表（capabilities/math/topics.TOPICS），level ∈ 基础/常规/提高；
- 公式只允许行内 $...$，禁止 \\[ \\] \\( \\) 块级（题目会直接流给前端）；
- answer 含关键步骤（3 行内），v1 不展示给学生，供数据完整性与后续判答扩展。

加载是校验式的：任何违规抛 ChatError(error_code="BANK_DATA_ERROR") 并带
定位信息——题库是教学质量的输入，脏数据必须当场暴露而不是上线后出事故。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nnnu.capabilities.math.pedagogy import DEFAULT_LEVEL, LEVELS
from nnnu.capabilities.math.topics import TOPICS
from nnnu.core import BaseTool, ChatError, ToolDefinition, ToolParameter, ToolResult

DEFAULT_BANK_PATH = Path(__file__).with_name("question_bank.json")

_REQUIRED_FIELDS = ("id", "topic", "level", "question", "answer")


@dataclass(frozen=True)
class BankItem:
    """题库中的一道题。"""

    id: str
    topic: str
    level: str
    question: str
    answer: str


@dataclass(frozen=True)
class QuestionBank:
    """已加载的题库（不可变）。"""

    items: tuple[BankItem, ...]

    def get(self, topic: str, level: str) -> list[BankItem]:
        """返回 (topic, level) 下全部题（按 JSON 顺序稳定）。"""
        return [item for item in self.items if item.topic == topic and item.level == level]


def load_question_bank(path: Path) -> QuestionBank:
    """校验式加载题库文件；违规抛 ChatError(BANK_DATA_ERROR)。"""
    if not path.is_file():
        raise ChatError(
            f"题库文件不存在: {path}", retryable=False, error_code="BANK_DATA_ERROR"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChatError(
            f"题库 JSON 解析失败（{path}）: {exc}", retryable=False, error_code="BANK_DATA_ERROR"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ChatError(
            f"题库结构错误（{path}）：需要 {{\"items\": [...]}}",
            retryable=False,
            error_code="BANK_DATA_ERROR",
        )
    if not payload["items"]:
        raise ChatError(
            f"题库为空（{path}）：items 不能为空列表",
            retryable=False,
            error_code="BANK_DATA_ERROR",
        )

    items: list[BankItem] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(payload["items"], 1):
        line_no = i + 1  # items[0] 大致在第 2 行（紧凑格式下仅作定位提示）
        if not isinstance(raw, dict):
            raise ChatError(
                f"题库第 {line_no} 条不是对象（{path}）",
                retryable=False,
                error_code="BANK_DATA_ERROR",
            )
        for field in _REQUIRED_FIELDS:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ChatError(
                    f"题库条目 {raw.get('id', f'第{i}条')} 缺少非空字段 {field}（{path}）",
                    retryable=False,
                    error_code="BANK_DATA_ERROR",
                )
        item_id = str(raw["id"]).strip()
        if item_id in seen_ids:
            raise ChatError(
                f"题库 id 重复: {item_id}（{path}）",
                retryable=False,
                error_code="BANK_DATA_ERROR",
            )
        seen_ids.add(item_id)
        topic = str(raw["topic"]).strip()
        if topic not in TOPICS:
            raise ChatError(
                f"题库条目 {item_id} 的 topic「{topic}」不在考点词表（{path}）",
                retryable=False,
                error_code="BANK_DATA_ERROR",
            )
        level = str(raw["level"]).strip()
        if level not in LEVELS:
            raise ChatError(
                f"题库条目 {item_id} 的 level「{level}」不在 {LEVELS}（{path}）",
                retryable=False,
                error_code="BANK_DATA_ERROR",
            )
        question = str(raw["question"]).strip()
        answer = str(raw["answer"]).strip()
        if "\\[" in question or "\\(" in question:
            raise ChatError(
                f"题库条目 {item_id} 的题目含块级公式 \\[ 或 \\(，只允许行内 $...$（{path}）",
                retryable=False,
                error_code="BANK_DATA_ERROR",
            )
        items.append(
            BankItem(id=item_id, topic=topic, level=level, question=question, answer=answer)
        )
    return QuestionBank(items=tuple(items))


class QuestionBankTool(BaseTool):
    """按 (topic, level) 取原创练习题。

    ``bank`` 供测试注入；生产按 ``bank_path`` 每次执行现装（改题库免重启，
    与 rag 工具同一约定）。同一 (topic, level) 多题时按轮转 cursor 依次
    取题不重复。
    """

    def __init__(self, bank: QuestionBank | None = None, *, bank_path: Path | None = None) -> None:
        self._bank = bank
        self._bank_path = bank_path if bank_path is not None else DEFAULT_BANK_PATH
        self._cursors: dict[tuple[str, str], int] = {}

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="question_bank",
            description=(
                "从考研数学（高等数学）题库取原创练习题：按考点与难度档位返回"
                "题目与答案。巩固阶段出变式题时调用。"
            ),
            parameters=[
                ToolParameter(
                    name="topic",
                    type="string",
                    required=True,
                    enum=list(TOPICS),
                    description="考点，取值见枚举（与教材知识库章节对应）",
                ),
                ToolParameter(
                    name="level",
                    type="string",
                    required=False,
                    enum=list(LEVELS),
                    description="难度档位：基础/常规/提高，缺省为常规",
                ),
            ],
        )

    def _get_bank(self) -> QuestionBank:
        if self._bank is None:
            self._bank = load_question_bank(self._bank_path)
        return self._bank

    async def execute(self, **kwargs: Any) -> ToolResult:
        topic = str(kwargs.get("topic", "")).strip()
        if not topic:
            return ToolResult(content="考点不能为空", success=False)
        if topic not in TOPICS:
            return ToolResult(
                content=f"未知考点「{topic}」，可选考点：{'、'.join(TOPICS)}", success=False
            )
        level = str(kwargs.get("level") or DEFAULT_LEVEL).strip()
        if level not in LEVELS:
            return ToolResult(
                content=f"未知难度档位「{level}」，可选：{'、'.join(LEVELS)}", success=False
            )
        try:
            bank = self._get_bank()
        except ChatError as exc:
            return ToolResult(content=f"题库不可用: {exc}", success=False)

        candidates = bank.get(topic, level)
        if not candidates:
            # 合法考点但库存缺口：success=True + item=None（与 rag 零命中语义一致）
            return ToolResult(
                content=f"题库暂未收录「{topic}·{level}」档位的题目",
                success=True,
                metadata={"item": None, "topic": topic, "level": level},
            )
        key = (topic, level)
        position = self._cursors.get(key, 0) % len(candidates)
        self._cursors[key] = position + 1
        item = candidates[position]
        return ToolResult(
            content=item.question,
            success=True,
            metadata={
                "item": {
                    "id": item.id,
                    "topic": item.topic,
                    "level": item.level,
                    "question": item.question,
                    "answer": item.answer,
                },
                "item_id": item.id,
                "position": position,
                "total": len(candidates),
            },
        )
