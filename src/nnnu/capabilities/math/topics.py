"""考研数学（高等数学）考点词表（Phase 2 W6）。

10 个主题与知识库语料（data/rag_corpus 的 10 篇文档）对齐：
- 题库（tools/bank/question_bank.json）的 topic 取值；
- 巩固阶段考点提取提示词（prompts.TOPIC_JSON_PROMPT）的候选表。

本模块零依赖：能力侧与工具侧都能 import，不产生循环。
"""

from __future__ import annotations

TOPICS: tuple[str, ...] = (
    "函数极限连续",
    "无穷小与夹逼准则",
    "洛必达法则",
    "导数与微分",
    "微分中值定理",
    "泰勒公式",
    "不定积分",
    "定积分与变限积分",
    "反常积分",
    "微分方程",
)
