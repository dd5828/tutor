"""个性化评测运行器（Phase 3 W8）：3 角色 × 10 轮连续对话。

与教学评测（run_evals.py）相反，本评测**必须激活工具**：新建 ToolRegistry
只注册 Read/WriteMemoryTool（临时 memory 目录）并注入能力，同一 session_id
连跑 10 轮，验证「第 2 轮后 AI 引用历史画像、难度与画像档位匹配」。

判分（LLM，temperature=0）两项：memory_referenced（诊断点名「上次你在…」
或按画像易错点/薄弱考点调整引导均算引用）、level_matched（提示粒度与讲解
深度与 expected_level 匹配）。通过线：计分轮两项各自 ≥80%。
CI 不跑本脚本（需真 key）；逻辑由 tests/unit/test_personalization_evals.py
假件覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nnnu.capabilities.math.capability import MathTutorCapability
from nnnu.capabilities.math.pedagogy import LEVELS, extract_json
from nnnu.core import StreamBus, StreamEvent, StreamEventType, TurnContext
from nnnu.runtime.registry import ToolRegistry
from nnnu.services.llm import LLMClient
from nnnu.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from tests.evals.run_evals import build_transcript

DEFAULT_CASES_PATH = Path(__file__).with_name("personalization.jsonl")
DEFAULT_OUT_PATH = Path(__file__).with_name("personalization_results.jsonl")
PASS_RATE = 0.8

RUBRIC = (
    "判分标准（两项，各判 true/false）：\n"
    "1. memory_referenced：AI 是否体现出记得这位学生的历史——诊断时点名"
    "「上次你在…卡在…」、或按画像易错点/薄弱考点调整引导，均算引用；"
    "全程无任何历史参考痕迹为 false；\n"
    "2. level_matched：AI 的提示粒度与讲解深度是否与画像档位"
    "（{expected_level}）匹配，明显偏难或明显偏易为 false。"
)


@dataclass
class PersonaTurn:
    """角色剧本中的一轮。"""

    question: str
    stuck_point: str
    replies: list[str] = field(default_factory=list)


@dataclass
class PersonaCase:
    """一个学生角色的完整剧本。"""

    id: str
    role: str
    session_id: str
    expected_level: str
    turns: list[PersonaTurn]


@dataclass
class PersonaTurnScore:
    """一轮的判分结果（第 1 轮三字段为 None：无历史不计分）。"""

    case_id: str
    turn_index: int
    memory_referenced: bool | None
    level_matched: bool | None
    passed: bool | None
    notes: str


def load_personalization_cases(path: Path) -> list[PersonaCase]:
    """加载并校验；违规 ValueError 带行号。"""
    if not path.is_file():
        raise ValueError(f"评测文件不存在: {path}")
    cases: list[PersonaCase] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            for key in ("id", "role", "session_id", "expected_level", "turns"):
                value = raw.get(key)
                if not isinstance(value, (str, list)) or not value:
                    raise ValueError(f"{path} 第 {line_no} 行缺少非空字段 {key}")
            expected_level = str(raw["expected_level"]).strip()
            if expected_level not in LEVELS:
                raise ValueError(
                    f"{path} 第 {line_no} 行 expected_level「{expected_level}」不在 {LEVELS}"
                )
            turns: list[PersonaTurn] = []
            raw_turns = raw["turns"]
            if not isinstance(raw_turns, list) or not raw_turns:
                raise ValueError(f"{path} 第 {line_no} 行 turns 必须是非空列表")
            for i, raw_turn in enumerate(raw_turns, 1):
                if not isinstance(raw_turn, dict):
                    raise ValueError(f"{path} 第 {line_no} 行第 {i} 轮不是对象")
                question = str(raw_turn.get("question", "")).strip()
                stuck_point = str(raw_turn.get("stuck_point", "")).strip()
                if not question or not stuck_point:
                    raise ValueError(f"{path} 第 {line_no} 行第 {i} 轮缺 question/stuck_point")
                replies = raw_turn.get("replies") or []
                turns.append(
                    PersonaTurn(
                        question=question,
                        stuck_point=stuck_point,
                        replies=[str(r).strip() for r in replies if isinstance(r, str)],
                    )
                )
            cases.append(
                PersonaCase(
                    id=str(raw["id"]).strip(),
                    role=str(raw["role"]).strip(),
                    session_id=str(raw["session_id"]).strip(),
                    expected_level=expected_level,
                    turns=turns,
                )
            )
    if not cases:
        raise ValueError(f"{path} 没有任何角色")
    return cases


async def _drive_turn(
    cap: MathTutorCapability, case: PersonaCase, turn: PersonaTurn
) -> tuple[list[StreamEvent], Exception | None]:
    """驱动一轮对话（stuck_point 答诊断反问，replies 依次答引导反问）。"""
    bus = StreamBus()
    ctx = TurnContext(user_message=turn.question, session_id=case.session_id)
    replies = [turn.stuck_point, *turn.replies]
    run_task = asyncio.create_task(cap.run(ctx, bus))

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
    return events, error


async def drive_role(
    cap: MathTutorCapability, case: PersonaCase, memory_dir: Path
) -> list[tuple[PersonaTurn, str, str]]:
    """连跑 10 轮；返回 [(turn, transcript, 本轮开始前的画像摘要)]。"""
    read_tool = ReadMemoryTool(memory_dir=memory_dir)
    results: list[tuple[PersonaTurn, str, str]] = []
    for turn in case.turns:
        summary = ""
        read_result = await read_tool.execute(session_id=case.session_id)
        if read_result.success:
            summary = str(read_result.metadata.get("summary", "")).strip()
        events, error = await _drive_turn(cap, case, turn)
        if error is not None:
            print(f"  ⚠ {case.id} 第 {len(results) + 1} 轮能力异常: {error}")
        transcript = build_transcript(events, [turn.stuck_point, *turn.replies])
        results.append((turn, transcript, summary))
    return results


def build_score_prompt(
    case: PersonaCase, turn_index: int, transcript: str, summary: str
) -> str:
    """判分提示词：角色设定 + 档位 + 截至本轮前的画像 + 对话全文 + 标准。"""
    return (
        f"学生角色：{case.role}\n"
        f"期望教学档位：{case.expected_level}\n"
        f"这是第 {turn_index} 轮对话。截至本轮开始前的画像摘要：\n"
        f"{summary or '（无历史）'}\n\n"
        f"对话全文：\n{transcript}\n\n"
        + RUBRIC.format(expected_level=case.expected_level)
        + "\n\n只输出 JSON，不要输出其他文字：\n"
        '{"memory_referenced": true或false, "level_matched": true或false, "notes": "一句话理由"}'
    )


async def score_turn(
    judge_llm: LLMClient,
    case: PersonaCase,
    turn_index: int,
    transcript: str,
    summary: str,
) -> PersonaTurnScore:
    """判一轮；第 1 轮豁免（无历史）；解析失败保守判不通过。"""
    if turn_index == 1:
        return PersonaTurnScore(
            case_id=case.id, turn_index=1,
            memory_referenced=None, level_matched=None, passed=None,
            notes="首轮无历史，不计分",
        )
    prompt = build_score_prompt(case, turn_index, transcript, summary)
    chunks: list[str] = []
    async for chunk in judge_llm.chat_stream(
        [{"role": "user", "content": prompt}], temperature=0.0
    ):
        chunks.append(chunk)
    data = extract_json("".join(chunks)) or {}
    memory_referenced = data.get("memory_referenced")
    level_matched = data.get("level_matched")
    notes = str(data.get("notes", ""))
    if not isinstance(memory_referenced, bool) or not isinstance(level_matched, bool):
        # 解析失败保守判不通过（宁严勿松）
        return PersonaTurnScore(
            case_id=case.id, turn_index=turn_index,
            memory_referenced=False, level_matched=False, passed=False,
            notes="判分解析失败",
        )
    return PersonaTurnScore(
        case_id=case.id, turn_index=turn_index,
        memory_referenced=memory_referenced, level_matched=level_matched,
        passed=memory_referenced and level_matched, notes=notes,
    )


def _write_results(out_path: Path, scores: list[PersonaTurnScore]) -> None:
    """落盘结果（同步函数，经 asyncio.to_thread 调用避开事件循环阻塞）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(asdict(score), ensure_ascii=False) + "\n")


def summarize(scores: list[PersonaTurnScore]) -> tuple[int, int, int]:
    """返回 (计分轮数, 引用数, 匹配数)。"""
    scored = [s for s in scores if s.passed is not None]
    return (
        len(scored),
        sum(1 for s in scored if s.memory_referenced),
        sum(1 for s in scored if s.level_matched),
    )


async def run(
    cases_path: Path,
    *,
    memory_dir: Path,
    cap_llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    limit: int | None = None,
    out_path: Path | None = None,
) -> list[PersonaTurnScore]:
    """真跑全部角色；返回 turn 级判分列表。"""
    cases = load_personalization_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]
    if cap_llm is None:
        cap_llm = LLMClient()
    if judge_llm is None:
        judge_llm = LLMClient()

    registry = ToolRegistry()
    registry.register(ReadMemoryTool(memory_dir=memory_dir))
    registry.register(WriteMemoryTool(memory_dir=memory_dir))
    capability = MathTutorCapability(llm=cap_llm, tools=registry)

    all_scores: list[PersonaTurnScore] = []
    for case in cases:
        print(f"角色 {case.id}（{case.expected_level}）开始，session={case.session_id}")
        rounds = await drive_role(capability, case, memory_dir)
        for i, (_, transcript, summary) in enumerate(rounds, 1):
            score = await score_turn(judge_llm, case, i, transcript, summary)
            all_scores.append(score)
            if score.passed is None:
                print(f"  第 {i} 轮：首轮豁免")
            else:
                print(
                    f"  第 {i} 轮：{'通过' if score.passed else '不通过'}"
                    f"（引用={score.memory_referenced}，匹配={score.level_matched}）{score.notes}"
                )

    total, refs, matched = summarize(all_scores)
    print(
        f"汇总：计分 {total} 轮，引用 {refs}（{refs / total:.1%}），"
        f"匹配 {matched}（{matched / total:.1%}）" if total else "汇总：无计分轮"
    )
    if total:
        passed = refs / total >= PASS_RATE and matched / total >= PASS_RATE
        print(f"通过线（各 ≥{PASS_RATE:.0%}）：{'达成' if passed else '未达成'}")
    if out_path is not None:
        await asyncio.to_thread(_write_results, out_path, all_scores)
        print(f"结果已写入 {out_path}")
    return all_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.evals.run_personalization_evals",
        description="个性化评测：3 角色 × 10 轮连续对话（真 key，验证画像引用与难度匹配）",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="评测数据 jsonl")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="画像目录（默认每次新建临时目录；指定可保留档案排查）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个角色")
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT_PATH), help=f"结果落盘路径（默认 {DEFAULT_OUT_PATH}）"
    )
    args = parser.parse_args(argv)

    memory_dir = Path(args.memory_dir) if args.memory_dir else Path(
        tempfile.mkdtemp(prefix="nnnu-persona-")
    )
    print(f"画像目录: {memory_dir}")
    try:
        asyncio.run(
            run(
                Path(args.cases),
                memory_dir=memory_dir,
                limit=args.limit,
                out_path=Path(args.out),
            )
        )
    except ValueError as exc:
        print(f"错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
