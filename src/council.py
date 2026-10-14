"""予想会議（M）・機械的集約（M-avg）・同予算シングル（S）を実行する（docs/COUNCIL.md）。

1つの run 番号ごとに、次の順で実行する。
1. M: 3役（球団動向・選手評価・検証）の独立提案（Round 0）→ 決められた順番で反論と修正（Round 1）→ 議長の最終予測
2. M-avg: Round 1 の3案を src/aggregate.py で機械的に集約する（LLM は呼ばない）
3. S: 条件B と同じプロンプトで独立に k 回予測し、同じ関数で集約する。k は、同じ run 番号の M の
   総トークン数（入力＋出力）の ±10% に入るように自動で決まる

最終予測（M・M-avg・S）だけを predictions/ に保存する。途中の案（Round 0・Round 1・S の各回）は、
採点の対象に混ざらないよう runs/council.jsonl にだけ残す（1行＝1回の LLM 呼び出し）。

使い方:
    uv run python -m src.council --predictions predictions/2026-10-21 --dry-run    # 見積もりだけ
    uv run python -m src.council --predictions predictions/2026-10-21              # run 1〜3
    uv run python -m src.council --predictions predictions/2026-10-21 --run-nos 2  # run 2 だけ
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from src.aggregate import aggregate
from src.models import MODELS, ModelSpec
from src.run import (
    CHARS_PER_TOKEN,
    MAX_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    PROMPTS,
    TODAY_PLACEHOLDER,
    Prompt,
    RetriesExhausted,
    RunLog,
    RunnerFn,
    _candidates_table,
    _retry_prompt,
    build_prompt,
    call_with_retry,
    extract_json,
    format_date,
    get_runner,
    today_jst,
)
from src.schema import NonEmptyStr, Prediction, PredictionOutput

MODEL_ID = "claude-sonnet-5-5"  # docs/COUNCIL.md 2章で固定（全役割・全条件で同じモデル）
COUNCIL_PROMPTS = PROMPTS / "council"
ROLES = ("teams", "players", "verifier")
CHAIR = "chair"
ROLE_LABELS = {"teams": "球団動向担当", "players": "選手評価担当", "verifier": "検証担当", CHAIR: "議長", "single": "シングル"}
# Round 1 の発言順（最初の発言者に引きずられていないかを見るため、run ごとに変える）
ORDERS: dict[int, tuple[str, str, str]] = {
    1: ("teams", "players", "verifier"),
    2: ("players", "verifier", "teams"),
    3: ("verifier", "teams", "players"),
}
BUDGET_TOLERANCE = 0.10  # S の総トークン数を M の ±10% にそろえる
MAX_SAMPLES = 50  # S の回数の上限（予算そろえの計算が壊れたときの安全装置）
DISCUSSION_PLACEHOLDER = "{{DISCUSSION}}"
JsonModel = type[BaseModel]


class CouncilTurn(BaseModel):
    """会議の1回の発言。LLM の出力をこれで検証する。"""

    model_config = ConfigDict(extra="forbid")

    argument: NonEmptyStr
    prediction: PredictionOutput


class TurnLog(BaseModel):
    """runs/council.jsonl の1行（1回の LLM 呼び出し、または S の予算そろえの結果）。"""

    run_id: str
    condition: str  # M / S
    run_no: int
    round: str  # round0 / round1 / final / sample / summary
    role: str  # teams / players / verifier / chair / single
    speaker_index: int | None  # Round 1 の発言順（0 始まり）。S では何回目か
    model_id: str
    prompt_sha256: str | None  # 実際に送ったプロンプト（議論を埋めた後）のハッシュ
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    attempts: int
    api_retries: int
    status: str  # ok / invalid_output / error
    error: str | None = None
    content: str | None = None  # 発言（argument）。S の summary では予算そろえの説明
    prediction_json: dict | None = None
    timestamp: str
    target_tokens: int | None = None  # S の summary: そろえる先（M の総トークン数）
    k: int | None = None  # S の summary: 有効な予測の数


def _sha(*parts: str) -> str:
    return hashlib.sha256("\n\n".join(parts).encode()).hexdigest()


@dataclass(frozen=True)
class CouncilPrompts:
    """会議のプロンプト。system は 共通の system.md＋会議の説明＋役割、user は 候補者一覧＋今回の発言の指示＋タスク。"""

    base_system: str
    common: str
    roles: dict[str, str]
    context: str
    rounds: dict[str, str]  # round0 / round1 / final
    task: str

    def system(self, role: str) -> str:
        return f"{self.base_system}\n\n{self.common}\n\n{self.roles[role]}"

    def user(self, round_name: str, discussion: str = "") -> str:
        text = self.rounds[round_name].replace(DISCUSSION_PLACEHOLDER, discussion)
        return f"{self.context}\n\n{text}\n\n{self.task}"

    @property
    def sha256(self) -> str:
        """テンプレート全体のハッシュ（runs.jsonl に記録する）。"""
        roles = [self.roles[r] for r in (*ROLES, CHAIR)]
        rounds = [self.rounds[r] for r in ("round0", "round1", "final")]
        return _sha(self.base_system, self.common, *roles, self.context, *rounds, self.task)


def build_council_prompts(
    prompts_dir: Path = PROMPTS, candidates: Path = Path("data/candidates.csv"), today: date | None = None
) -> CouncilPrompts:
    today_text = format_date(today or today_jst())

    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8").replace(TODAY_PLACEHOLDER, today_text)

    council = prompts_dir / "council"
    schema = json.dumps(CouncilTurn.model_json_schema(), ensure_ascii=False, indent=1)
    return CouncilPrompts(
        base_system=read(prompts_dir / "system.md"),
        common=read(council / "common.md"),
        roles={r: read(council / f"{r}.md") for r in (*ROLES, CHAIR)},
        context=read(prompts_dir / "context_B.md").replace("{{CANDIDATES}}", _candidates_table(candidates)),
        rounds={r: read(council / f"{r}.md") for r in ("round0", "round1", "final")},
        task=read(prompts_dir / "task.md").replace("{{SCHEMA}}", schema),
    )


def _statement(role: str, round_name: str, turn: CouncilTurn) -> str:
    pred = turn.prediction.model_dump_json()  # 改行なしで入力トークンを抑える
    return f"### {ROLE_LABELS[role]}（{round_name}）\n\n発言:\n{turn.argument}\n\n予測:\n{pred}"


@dataclass
class Caller:
    """LLM を呼び、JSON を検証する。スキーマ違反なら、エラーを添えて1回だけ出し直す（src/run.py と同じ）。"""

    spec: ModelSpec
    runner: RunnerFn
    sleep: Callable[[float], None] = time.sleep

    def __call__(
        self, model: JsonModel, system: str, user: str, *, run_id: str, condition: str, run_no: int, round_name: str, role: str, index: int | None
    ) -> tuple[BaseModel | None, TurnLog]:
        t0, now = time.monotonic(), datetime.now(UTC)
        stats = {"in": 0, "out": 0, "retries": 0, "attempts": 0}
        result: BaseModel | None = None
        status, error, text = "invalid_output", None, ""

        def call(u: str) -> str:
            res, retries = call_with_retry(self.runner, (self.spec.model_id, system, u, MAX_OUTPUT_TOKENS), self.sleep)
            stats["retries"] += retries
            stats["in"] += res.input_tokens
            stats["out"] += res.output_tokens
            return res.text

        try:
            text = call(user)
            for retry in (False, True):
                stats["attempts"] += 1
                try:
                    result = model.model_validate_json(extract_json(text))
                    status, error = "ok", None
                    break
                except ValidationError as e:
                    error = f"{e.error_count()}件のスキーマ違反: {str(e)[:2000]}"
                    if not retry:
                        text = call(_retry_prompt(user, error, text))
        except RetriesExhausted as e:
            stats["retries"] += e.retries
            status, error = "error", str(e)[:2000]
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {str(e)[:2000]}"

        pred = result.prediction if isinstance(result, CouncilTurn) else result
        log = TurnLog(
            run_id=run_id,
            condition=condition,
            run_no=run_no,
            round=round_name,
            role=role,
            speaker_index=index,
            model_id=self.spec.model_id,
            prompt_sha256=_sha(system, user),
            input_tokens=stats["in"],
            output_tokens=stats["out"],
            cost_usd=round(self.spec.cost_usd(stats["in"], stats["out"]), 6),
            latency_ms=int((time.monotonic() - t0) * 1000),
            attempts=stats["attempts"],
            api_retries=stats["retries"],
            status=status,
            error=error,
            content=result.argument if isinstance(result, CouncilTurn) else (None if result else text[:20000] or None),
            prediction_json=pred.model_dump(mode="json") if pred is not None else None,
            timestamp=now.isoformat(),
        )
        return result, log


@dataclass
class Outcome:
    """1つの条件の1回の実行の結果。"""

    condition: str
    run_id: str
    run_no: int
    started: datetime
    prompt_sha256: str
    turns: list[TurnLog] = field(default_factory=list)
    output: PredictionOutput | None = None
    round1: list[PredictionOutput] = field(default_factory=list)  # M のみ。M-avg の材料（発言順）
    error: str | None = None
    duration_sec: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)


def _run_id(condition: str, model_id: str, run_no: int, started: datetime) -> str:
    return f"{condition}-{model_id}-{run_no:02d}-{started.strftime('%Y%m%dT%H%M%SZ')}"


def run_council(caller: Caller, prompts: CouncilPrompts, run_no: int) -> Outcome:
    """M の1回分。どれかの発言がスキーマ検証に通らなければ、そこで打ち切る。"""
    started, t0 = datetime.now(UTC), time.monotonic()
    out = Outcome("M", _run_id("M", caller.spec.model_id, run_no, started), run_no, started, prompts.sha256)

    def speak(role: str, round_name: str, discussion: str = "", index: int | None = None) -> CouncilTurn | None:
        res, log = caller(
            CouncilTurn,
            prompts.system(role),
            prompts.user(round_name, discussion),
            run_id=out.run_id,
            condition="M",
            run_no=run_no,
            round_name=round_name,
            role=role,
            index=index,
        )
        out.turns.append(log)
        if res is None:
            out.error = f"{ROLE_LABELS[role]}（{round_name}）: {log.error}"
        return res if isinstance(res, CouncilTurn) else None

    round0: dict[str, CouncilTurn] = {}
    for role in ROLES:
        turn = speak(role, "round0")
        if turn is None:
            return _finish(out, t0)
        round0[role] = turn
    proposals = "\n\n".join(_statement(r, "Round 0", round0[r]) for r in ROLES)

    round1: list[tuple[str, CouncilTurn]] = []
    for i, role in enumerate(ORDERS[run_no]):
        earlier = [_statement(r, "Round 1", t) for r, t in round1]
        discussion = "\n\n".join([proposals, *earlier])
        turn = speak(role, "round1", discussion, i)
        if turn is None:
            return _finish(out, t0)
        round1.append((role, turn))
    out.round1 = [t.prediction for _, t in round1]

    final = speak(CHAIR, "final", "\n\n".join(_statement(r, "Round 1", t) for r, t in round1))
    if final is not None:
        out.output = final.prediction
    return _finish(out, t0)


def _finish(out: Outcome, t0: float) -> Outcome:
    out.duration_sec = round(time.monotonic() - t0, 2)
    return out


def single_should_continue(used: int, per_call: list[int], target: int, tolerance: float = BUDGET_TOLERANCE) -> bool:
    """S をもう1回呼ぶか。下限に届くまで呼び、次の1回（これまでの平均で見積もる）で上限を超えるなら止める。"""
    if used >= target * (1 - tolerance) or len(per_call) >= MAX_SAMPLES:
        return False
    return not per_call or used + statistics.fmean(per_call) <= target * (1 + tolerance)


def run_single(caller: Caller, prompt: Prompt, run_no: int, target_tokens: int) -> Outcome:
    """S の1回分。条件B と同じプロンプトで独立に予測し、総トークン数を target_tokens の ±10% にそろえて集約する。"""
    started, t0 = datetime.now(UTC), time.monotonic()
    out = Outcome("S", _run_id("S", caller.spec.model_id, run_no, started), run_no, started, prompt.sha256)
    samples: list[PredictionOutput] = []
    per_call: list[int] = []
    while single_should_continue(sum(per_call), per_call, target_tokens):
        res, log = caller(
            PredictionOutput,
            prompt.system,
            prompt.user,
            run_id=out.run_id,
            condition="S",
            run_no=run_no,
            round_name="sample",
            role="single",
            index=len(per_call),
        )
        out.turns.append(log)
        per_call.append(log.input_tokens + log.output_tokens)
        if log.status == "error" and log.input_tokens == 0:  # API が応答しない: 予算が減らないので打ち切る
            out.error = f"S の {len(per_call)} 回目: {log.error}"
            break
        if isinstance(res, PredictionOutput):
            samples.append(res)
    used = sum(per_call)
    ratio = used / target_tokens if target_tokens else 0.0
    matched = abs(ratio - 1) <= BUDGET_TOLERANCE
    if samples:
        out.output = aggregate(samples)
    elif out.error is None:
        out.error = "S の予測がすべてスキーマ検証に通りませんでした"
    out.turns.append(
        TurnLog(
            run_id=out.run_id,
            condition="S",
            run_no=run_no,
            round="summary",
            role="single",
            speaker_index=None,
            model_id=caller.spec.model_id,
            prompt_sha256=prompt.sha256,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
            attempts=0,
            api_retries=0,
            status="ok" if matched and samples else "budget_mismatch" if samples else "error",
            content=f"呼び出し {len(per_call)} 回、有効な予測 {len(samples)} 個、{used} / {target_tokens} トークン（{ratio:.1%}）",
            timestamp=datetime.now(UTC).isoformat(),
            target_tokens=target_tokens,
            k=len(samples),
        )
    )
    return _finish(out, t0)


def save_prediction(predictions_dir: Path, condition: str, run_id: str, model_id: str, output: PredictionOutput) -> str:
    pred = Prediction(run_id=run_id, model_id=model_id, condition=condition, output=output)
    path = predictions_dir / condition / model_id / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pred.model_dump_json(indent=1), encoding="utf-8")
    return str(path)


def run_log(spec: ModelSpec, out: Outcome, path: str | None) -> RunLog:
    """runs/runs.jsonl の1行。本体の実行と同じ形式。"""
    calls = [t for t in out.turns if t.round != "summary"]
    tokens_in = sum(t.input_tokens for t in calls)
    tokens_out = sum(t.output_tokens for t in calls)
    return RunLog(
        run_id=out.run_id,
        model_id=spec.model_id,
        provider=spec.provider.value,
        condition=out.condition,
        run_no=out.run_no,
        prompt_sha256=out.prompt_sha256,
        started_at=out.started.isoformat(),
        duration_sec=out.duration_sec,
        attempts=sum(t.attempts for t in calls),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        reasoning_tokens=None,
        cost_usd=round(spec.cost_usd(tokens_in, tokens_out), 6),
        tool_calls=0,
        api_retries=sum(t.api_retries for t in calls),
        status="ok" if out.output is not None else "error",
        error=out.error,
        prediction_path=path,
    )


TYPICAL_STATEMENT_TOKENS = 6_000  # 目安の見積もりで、1つの発言（argument＋予測の JSON）の出力とみなす量（実測で見直す）


def estimate_cost(
    spec: ModelSpec, prompts: CouncilPrompts, n_runs: int, statement_tokens: int = MAX_OUTPUT_TOKENS, attempts: int = MAX_ATTEMPTS
) -> tuple[float, float]:
    """見積もり (M の n_runs 回分, S の n_runs 回分)。M-avg は LLM を呼ばないので 0。

    既定値は src/run.py と同じ上限: 出力は max_output_tokens を使い切り、すべての発言で出し直しが起きた場合。
    議論に入る発言は、1つあたり statement_tokens とみなす。
    S は M のトークン数の +10% まで使う。上限では単価を入力と出力の高い方で、目安では M と同じ比率で計算する。
    """

    def tokens(text: str) -> int:
        return int(len(text) / CHARS_PER_TOKEN)

    # (役割, 発言, 議論に入る発言の数)
    calls = [(r, "round0", 0) for r in ROLES] + [(r, "round1", 3 + i) for i, r in enumerate(ROLES)] + [(CHAIR, "final", 3)]
    m_in = m_out = 0
    for role, round_name, prior in calls:
        first_in = tokens(prompts.system(role) + prompts.user(round_name)) + prior * statement_tokens
        m_in += first_in + (first_in + statement_tokens) * (attempts - 1)  # 出し直しには前回の出力が入る
        m_out += statement_tokens * attempts
    m_cost = spec.cost_usd(m_in, m_out)
    if attempts == MAX_ATTEMPTS and statement_tokens == MAX_OUTPUT_TOKENS:
        s_cost = (m_in + m_out) * (1 + BUDGET_TOLERANCE) * max(spec.input_usd_per_mtok, spec.output_usd_per_mtok) / 1_000_000
    else:
        s_cost = m_cost * (1 + BUDGET_TOLERANCE)
    return m_cost * n_runs, s_cost * n_runs


def _append(path: Path, lines: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.model_dump_json() + "\n")


def _parse_run_nos(text: str) -> list[int]:
    nos = [int(x) for x in text.split(",") if x.strip()]
    bad = [n for n in nos if n not in ORDERS]
    if bad or not nos:
        raise argparse.ArgumentTypeError(f"run 番号は {sorted(ORDERS)} から選んでください（指定: {text}）")
    return nos


def main(argv: list[str] | None = None, runner: RunnerFn | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID, choices=sorted(MODELS))
    ap.add_argument("--run-nos", type=_parse_run_nos, default=[1, 2, 3], help="実行する run 番号（カンマ区切り。発言順が決まる）")
    ap.add_argument("--predictions", type=Path, required=True, help="保存先（事前登録の回のディレクトリ）")
    ap.add_argument("--runs", type=Path, default=Path("runs/runs.jsonl"))
    ap.add_argument("--council-log", type=Path, default=Path("runs/council.jsonl"))
    ap.add_argument("--budget", type=float, help="予算上限（USD）。省略時は環境変数 BUDGET_USD")
    ap.add_argument("--dry-run", action="store_true", help="見積もりだけ表示して終了")
    args = ap.parse_args(argv)

    spec = MODELS[args.model]
    prompts = build_council_prompts()
    single_prompt = build_prompt("B")
    m_cost, s_cost = estimate_cost(spec, prompts, len(args.run_nos))
    estimate = m_cost + s_cost
    typical = sum(estimate_cost(spec, prompts, len(args.run_nos), TYPICAL_STATEMENT_TOKENS, attempts=1))
    print(
        f"{spec.model_id} 予想会議 run {args.run_nos}: 見積もり上限 ${estimate:.2f}（M ${m_cost:.2f}＋S ${s_cost:.2f}）、"
        f"目安 ${typical:.2f}（発言1つ {TYPICAL_STATEMENT_TOKENS} トークン、出し直しなし）"
        f"（実行日 {format_date(today_jst())}、council prompt {prompts.sha256[:12]}、S prompt {single_prompt.sha256[:12]}）"
    )
    if args.dry_run:
        return 0

    budget = args.budget if args.budget is not None else float(os.environ.get("BUDGET_USD") or "nan")
    if not budget == budget:  # NaN = 未設定
        print("予算が未設定です。--budget か環境変数 BUDGET_USD を指定してください", file=sys.stderr)
        return 1
    if estimate > budget:
        print(f"見積もり上限 ${estimate:.2f} が予算 ${budget:.2f} を超えるため停止します", file=sys.stderr)
        return 1

    caller = Caller(spec, runner or get_runner(spec.provider), sleep)
    failed = 0
    for run_no in args.run_nos:
        m = run_council(caller, prompts, run_no)
        _append(args.council_log, m.turns)
        m_path = save_prediction(args.predictions, "M", m.run_id, spec.model_id, m.output) if m.output else None
        _append(args.runs, [run_log(spec, m, m_path)])
        print(f"  run {run_no} M: {'ok' if m.output else 'error'} tokens={m.tokens} ${run_log(spec, m, m_path).cost_usd:.4f} {m.duration_sec}s")
        if m.output is None:
            print(f"    {m.error}", file=sys.stderr)
            failed += 1
            continue  # M がなければ、M-avg も S の予算も決まらない

        avg = Outcome("M-avg", _run_id("M-avg", spec.model_id, run_no, m.started), run_no, m.started, prompts.sha256)
        avg.output = aggregate(m.round1)
        avg_path = save_prediction(args.predictions, "M-avg", avg.run_id, spec.model_id, avg.output)
        _append(args.runs, [run_log(spec, avg, avg_path)])

        s = run_single(caller, single_prompt, run_no, m.tokens)
        _append(args.council_log, s.turns)
        s_path = save_prediction(args.predictions, "S", s.run_id, spec.model_id, s.output) if s.output else None
        _append(args.runs, [run_log(spec, s, s_path)])
        summary = s.turns[-1]
        print(f"  run {run_no} S: {summary.status} {summary.content} ${run_log(spec, s, s_path).cost_usd:.4f} {s.duration_sec}s")
        if s.error:
            print(f"    {s.error}", file=sys.stderr)
        failed += s.output is None or summary.status != "ok"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
