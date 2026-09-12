"""评测集运行器：跑 cases.jsonl，LLM 按教学法评分标准判分（教学法文档 §8）。

用法（需 NNNU_LLM_API_KEY）：
    python -m tests.evals.run_evals
    python -m tests.evals.run_evals --limit 5 --out tests/evals/results.jsonl

流程：
1. 每条 case 用 MathTutorCapability 驱动一轮真实对话——学生回复按 case 预编排
   （stuck_point 回答诊断反问，replies 依次回答引导反问，耗尽后投递空输入）；
2. 教学动作全文（AI 输出 + 反问 + 学生回复）交给 LLM 按评分标准打结构化分；
3. 通过线：无严重违规（未引导直接给完整答案）且总分 ≥ 8/13（教学法文档 §8.3）。

CI 不跑本脚本（需真 key）；其逻辑由 tests/unit/test_evals.py 用假 LLM 覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nnnu.capabilities.math.capability import MathTutorCapability
from nnnu.capabilities.math.pedagogy import LEVELS, extract_json
from nnnu.core import StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.services.llm import LLMClient

# 期望行为词表（case 的 expected 只能取这些值；判分提示词据此核对）
EXPECTED_KEYS = {
    "确认题意",
    "引导而非给答案",
    "错因定位=概念",
    "错因定位=公式",
    "错因定位=方法",
    "错因定位=计算",
    "降级阶梯",
    "讲解四段",
    "变式题",
    "LaTeX格式",
}

# 通过线（教学法文档 §8.3）：满分 13，总分 ≥ 8 且无严重违规
MAX_TOTAL = 13
PASS_TOTAL_THRESHOLD = 8
SCORE_KEYS = ("one_step", "error_layer", "ladder", "explain", "variant", "latex")

RUBRIC = (
    "评分标准（满分 13）：\n"
    "1. 未引导直接给完整答案（-10，严重违规，命中即不通过）。"
    "注意：〔进入阶段：讲解〕之后的完整讲解是预期行为，不属违规；"
    "本项只罚「引导阶段（诊断/引导）里直接给出完整答案或关键解法」；\n"
    "2. 一次只给一步提示并反问等作答（0-3）；\n"
    "3. 学生答错时先定位错因层再引导（0-3）；\n"
    "4. 降级阶梯正确：先最小提示后更细提示、最后才完整讲解（0-2）；\n"
    "5. 讲解含考点/思路/易错点三要素（0-2）；\n"
    "6. 巩固阶段出变式题（0-2）；\n"
    "7. 数学公式用 LaTeX 格式（0-1）。"
)

SCORE_JSON_EXAMPLE = (
    '{"violations": [{"rule": "直接给完整答案", "evidence": "原文摘录"}], '
    '"scores": {"one_step": 0, "error_layer": 0, "ladder": 0, '
    '"explain": 0, "variant": 0, "latex": 0}, "notes": "一句评语"}'
)


@dataclass
class EvalCase:
    """一条评测 case（教学法文档 §8.2）。"""

    id: str
    level: str
    question: str
    stuck_point: str
    expected: list[str]
    replies: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    """一条 case 的判分结果。"""

    case_id: str
    total: int
    violations: list[str]
    notes: str
    passed: bool


def load_cases(path: Path) -> list[EvalCase]:
    """加载 cases.jsonl；字段/词表非法直接报错（数据完整性优先）。"""
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = json.loads(line)
            required = ("id", "level", "question", "stuck_point", "expected")
            missing = [k for k in required if k not in raw]
            if missing:
                raise ValueError(f"{path}:{line_no} 缺字段 {missing}")
            level = str(raw["level"])
            if level not in LEVELS:
                raise ValueError(f"{path}:{line_no} level={level} 非法（{LEVELS}）")
            expected = [str(k) for k in raw["expected"]]
            unknown = [k for k in expected if k not in EXPECTED_KEYS]
            if unknown:
                raise ValueError(f"{path}:{line_no} 未知期望行为 {unknown}")
            cases.append(
                EvalCase(
                    id=str(raw["id"]),
                    level=level,
                    question=str(raw["question"]),
                    stuck_point=str(raw["stuck_point"]),
                    expected=expected,
                    replies=[str(r) for r in raw.get("replies", [])],
                )
            )
    return cases


async def run_case(
    cap: MathTutorCapability,
    case: EvalCase,
) -> tuple[str, Exception | None]:
    """驱动一轮对话；返回（教学动作全文, 运行异常（正常为 None））。"""
    bus = StreamBus()
    replies = [case.stuck_point, *case.replies]
    run_task = asyncio.create_task(
        cap.run(TurnContext(user_message=case.question), bus)
    )

    async def consume() -> list[StreamEvent]:
        events: list[StreamEvent] = []
        async for event in bus.subscribe():
            events.append(event)
            if event.type == StreamEventType.WAIT_FOR_INPUT:
                bus.submit_input(replies.pop(0) if replies else "")
        return events

    consumer = asyncio.create_task(consume())
    error: Exception | None = None
    try:
        await run_task
    except Exception as exc:
        error = exc
    bus.mark_closed()
    events = await consumer
    return build_transcript(events, replies_snapshot(case)), error


def replies_snapshot(case: EvalCase) -> list[str]:
    """学生回复的完整序列（与 run_case 内投递顺序一致，供对话还原）。"""
    return [case.stuck_point, *case.replies]


def build_transcript(events: list[StreamEvent], replies: list[str]) -> str:
    """把事件流还原成对话全文：阶段标记 / AI 输出 / 反问 / 学生回复。

    阶段标记是判分的关键上下文：讲解阶段的完整讲解是预期行为，
    没有阶段信息判分 LLM 会把它误判成"直接给答案"违规。
    """
    lines: list[str] = []
    reply_iter = iter(replies)
    for event in events:
        if event.type == StreamEventType.STAGE_START:
            lines.append(f"〔进入阶段：{event.stage}〕")
        elif event.type == StreamEventType.CONTENT:
            lines.append(f"[AI] {event.content}")
        elif event.type == StreamEventType.WAIT_FOR_INPUT:
            lines.append(f"[问] {event.content}")
            lines.append(f"[学生] {next(reply_iter, '') or '(未作答)'}")
    return "\n".join(lines)


def build_score_prompt(case: EvalCase, transcript: str) -> str:
    return (
        f"下面是一段考研数学 AI 辅导对话（〔进入阶段：X〕=阶段边界，"
        f"[问]=AI 的反问，[学生]=学生回复）：\n"
        f"{transcript}\n\n"
        f"本 case 期望行为：{'、'.join(case.expected)}\n\n"
        f"{RUBRIC}\n"
        f"按评分标准打分，只输出 JSON，不要输出其他文字：\n{SCORE_JSON_EXAMPLE}"
    )


async def score_case(
    llm: LLMClient,
    case: EvalCase,
    transcript: str,
) -> CaseScore:
    """LLM 按评分标准判分；解析失败保守判不通过（评分体系宁严勿松）。"""
    chunks: list[str] = []
    async for chunk in llm.chat_stream(
        [{"role": "user", "content": build_score_prompt(case, transcript)}],
        temperature=0.0,
    ):
        chunks.append(chunk)
    data = extract_json("".join(chunks)) or {}
    scores = data.get("scores") or {}
    total = 0
    for key in SCORE_KEYS:
        try:
            total += max(0, min(int(scores.get(key, 0)), MAX_TOTAL))
        except (TypeError, ValueError):
            continue
    violations = [
        str(v.get("rule", "未注明规则"))
        for v in data.get("violations", [])
        if isinstance(v, dict)
    ]
    if not data:
        violations = ["评分解析失败"]
    return CaseScore(
        case_id=case.id,
        total=total,
        violations=violations,
        notes=str(data.get("notes", "")) if data else "LLM 未输出可解析 JSON",
        passed=not violations and total >= PASS_TOTAL_THRESHOLD,
    )


def format_result(score: CaseScore) -> str:
    verdict = "通过" if score.passed else "不通过"
    notes = f"｜{score.notes}" if score.notes else ""
    violations = f"｜违规: {'; '.join(score.violations)}" if score.violations else ""
    return f"{verdict}｜总分 {score.total}/{MAX_TOTAL}{violations}{notes}"


async def run(
    cases_path: Path,
    *,
    limit: int | None = None,
    out_path: Path | None = None,
) -> list[CaseScore]:
    """跑全部 case，打印汇总，可选把结果写 JSONL。"""
    cases = load_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]
    llm = LLMClient()
    cap = MathTutorCapability(llm=llm)  # 用量走全局 tracker（JSON 日志），便于核成本
    results: list[CaseScore] = []
    for index, case in enumerate(cases, 1):
        transcript, error = await run_case(cap, case)
        if error is not None:
            transcript = f"[运行错误] {error}\n{transcript}"
        score = await score_case(llm, case, transcript)
        results.append(score)
        print(f"[{index}/{len(cases)}] {case.id}（{case.level}） {format_result(score)}")
        if error is not None:
            print(f"  ↳ 运行异常: {error}")
    passed = sum(1 for r in results if r.passed)
    print(f"通过率: {passed}/{len(results)}")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for score in results:
                f.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")
        print(f"结果已写入 {out_path}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="考研数学教学评测集运行器")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).parent / "cases.jsonl",
        help="评测数据（默认 tests/evals/cases.jsonl）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（抽样）")
    parser.add_argument("--out", type=Path, default=None, help="结果写入 JSONL 路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(run(args.cases, limit=args.limit, out_path=args.out))
    except Exception as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
