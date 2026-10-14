"""予測を実行して predictions/ に保存し、実行ログを runs/runs.jsonl に記録する。

現在の対応: 条件 A（情報なし）・B（候補者一覧をコンテキストに入れる）・C（各社 API の Web 検索）・
D（Web 検索＋draft-db のツール＋自己検証）。H は未実装。
条件 C は2段階: Web 検索ありで調査メモを書かせ、検索なしでメモから JSON を出力させる。
条件 D は3段階: 調査（Web 検索＋draft-db）→ 整形 → 自己検証（draft-db）（docs/PLAN.md 3.2）。

ツールを使わない段階の出力は各社の JSON モードで受け取り、PredictionOutput（src/schema.py）で検証する。
スキーマ違反のときは、エラー内容を添えて1回だけ出し直しを求める（attempts に記録）。

使い方:
    uv run python -m src.run --condition B --model gemini-3.8-flash --n 1
    uv run python -m src.run --condition B --model gemini-3.8-flash --n 1 --dry-run   # 見積もりだけ
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.candidates import load_candidates
from src.draft_db_mcp.tools import DraftDB
from src.models import MODELS, ModelSpec, Provider
from src.runners.base import MAX_SEARCHES, MAX_TOOL_TURNS, RunnerResult, ToolBox
from src.schema import Prediction, PredictionOutput

PROMPTS = Path("prompts")
CONDITIONS = ("A", "B", "C", "D")
MAX_OUTPUT_TOKENS = 32_000
MAX_ATTEMPTS = 2  # 初回＋スキーマ違反時の出し直し1回
CHARS_PER_TOKEN = 1.0  # 日本語の見積もり用（実測で見直す）
API_RETRY_DELAYS = (10.0, 30.0, 90.0)  # 混雑（503）やレート制限など、API の例外時に待つ秒数

RunnerFn = Callable[..., RunnerResult]  # (model_id, system, user, max_output_tokens, web_search=False)


class RetriesExhausted(Exception):
    """再試行を使い切った。最後の例外を cause に持つ。"""

    def __init__(self, retries: int, last: Exception) -> None:
        super().__init__(f"{type(last).__name__}: {last}")
        self.retries = retries


def call_with_retry(
    runner: RunnerFn,
    args: tuple[str, str, str, int],
    sleep: Callable[[float], None] = time.sleep,
    web_search: bool = False,
    tools: ToolBox | None = None,
) -> tuple[RunnerResult, int]:
    """API の例外は待ってから再試行する。(結果, 再試行回数) を返し、再試行し尽くしたら RetriesExhausted を投げる。"""
    extra = {"tools": tools} if tools is not None else {}  # ツールを使わない条件では、ランナーに tools を渡さない
    for retries, delay in enumerate((*API_RETRY_DELAYS, None)):
        try:
            return runner(*args, web_search=web_search, **extra), retries
        except Exception as e:
            if delay is None:
                raise RetriesExhausted(retries, e) from e
            sleep(delay)
    raise AssertionError("unreachable")


NOTES_PLACEHOLDER = "{{NOTES}}"
TODAY_PLACEHOLDER = "{{TODAY}}"
JST = timezone(timedelta(hours=9), "JST")


def today_jst() -> date:
    return datetime.now(JST).date()


def format_date(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"
DRAFT_PLACEHOLDER = "{{DRAFT}}"


@dataclass(frozen=True)
class Stage:
    """JSON を出力する段階の前後に置く段階（調査・自己検証）のプロンプトと、使うツール。"""

    system: str
    user: str
    web_search: bool = False
    use_db: bool = False  # draft-db のツール（search_players・get_player）を関数として渡す


@dataclass(frozen=True)
class Prompt:
    """1回の実行のプロンプト。

    条件 A・B は system・user だけ。条件 C は research（調査）→ 整形、条件 D はさらに verify（自己検証）を足す。
    research があるとき user は NOTES_PLACEHOLDER を含み、調査メモで置き換えてから使う。
    verify の user は NOTES_PLACEHOLDER と DRAFT_PLACEHOLDER（整形した JSON）を含む。
    """

    system: str
    user: str
    research: Stage | None = None
    verify: Stage | None = None

    @property
    def uses_db(self) -> bool:
        return any(s.use_db for s in (self.research, self.verify) if s)

    @property
    def sha256(self) -> str:
        parts = [self.system, self.user]
        for stage in (self.research, self.verify):
            if stage:
                parts += [stage.system, stage.user, f"web_search={stage.web_search} use_db={stage.use_db}"]
        return hashlib.sha256("\n\n".join(parts).encode()).hexdigest()


def _candidates_table(path: Path) -> str:
    rows = load_candidates(path)
    return "\n".join(f"{c.category.value},{c.name},{c.team},{c.position or ''}" for c in rows)


def build_prompt(
    condition: str,
    prompts_dir: Path = PROMPTS,
    candidates: Path = Path("data/candidates.csv"),
    today: date | None = None,
) -> Prompt:
    """条件のプロンプトを組み立てる。{{TODAY}} は実行日（日本時間）で置き換える。

    日付が変わるとプロンプトのハッシュも変わる（ハッシュは runs.jsonl に記録される）。
    """
    if condition not in CONDITIONS:
        raise ValueError(f"未対応の条件です: {condition}（対応: {', '.join(CONDITIONS)}）")
    today_text = format_date(today or today_jst())

    def read(name: str) -> str:
        return (prompts_dir / name).read_text(encoding="utf-8").replace(TODAY_PLACEHOLDER, today_text)

    schema = json.dumps(PredictionOutput.model_json_schema(), ensure_ascii=False, indent=1)
    task = read("task.md").replace("{{SCHEMA}}", schema)
    system = read("system.md")
    if condition == "A":
        return Prompt(system=system, user=task)
    if condition == "B":
        return Prompt(system=system, user=read("context_B.md").replace("{{CANDIDATES}}", _candidates_table(candidates)) + "\n\n" + task)
    formatting = read("context_C.md") + "\n\n" + task  # 調査メモから JSON を出力する（C・D 共通）
    if condition == "C":
        return Prompt(system=system, user=formatting, research=Stage(read("research_system.md"), read("research_C.md"), web_search=True))
    return Prompt(
        system=system,
        user=formatting,
        research=Stage(read("research_system.md"), read("research_D.md"), web_search=True, use_db=True),
        verify=Stage(system, read("verify_D.md") + "\n\n" + task, use_db=True),
    )


# 検索結果は入力トークンとして課金される。検索1回あたりの上乗せの目安。
# Claude は pause_turn から再開するたびに、それまでの検索結果を入力として送り直すため大きい（2段階の試行で 59.6万 / 10回）
SEARCH_RESULT_TOKENS = 60_000


# 関数呼び出しを1往復するたびに増える入力の目安（それまでの会話を送り直す分を含む。実測で見直す）
TOOL_TURN_TOKENS = 20_000


def estimate_max_cost(spec: ModelSpec, prompt: Prompt, n: int) -> float:
    """見積もりの上限。出力は max_output_tokens を使い切り、全実行で出し直しが起きた場合。

    調査メモ（条件 C・D）は整形・自己検証の入力に入るので、その入力にも max_output_tokens を足す。
    """

    def tokens(text: str) -> int:
        return int(len(text) / CHARS_PER_TOKEN)

    def stage_cost(stage: Stage, extra_input: int) -> float:
        input_tokens = tokens(stage.system + stage.user) + extra_input
        searches = MAX_SEARCHES if stage.web_search else 0
        input_tokens += searches * SEARCH_RESULT_TOKENS + (MAX_TOOL_TURNS * TOOL_TURN_TOKENS if stage.use_db else 0)
        return spec.cost_usd(input_tokens, MAX_OUTPUT_TOKENS, searches)

    notes = MAX_OUTPUT_TOKENS if prompt.research else 0
    per_attempt = spec.cost_usd(tokens(prompt.system + prompt.user) + notes + MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
    total = per_attempt * MAX_ATTEMPTS
    if prompt.research:
        total += stage_cost(prompt.research, 0)
    if prompt.verify:
        total += stage_cost(prompt.verify, notes + MAX_OUTPUT_TOKENS)  # 調査メモと確認前の JSON が入る
        total += per_attempt  # 自己検証の出力がスキーマに違反したときの出し直し
    return total * n


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(text: str) -> str:
    """コードブロックや前後の文を取り除き、最初の { から最後の } までを返す。"""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class RunLog(BaseModel):
    """runs/runs.jsonl の1行（AGENTS.md ルール3）。"""

    run_id: str
    model_id: str
    provider: str
    condition: str
    run_no: int  # seed の代わり（各社 API は seed を保証しないため）
    prompt_sha256: str
    started_at: str
    duration_sec: float
    attempts: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    cost_usd: float
    tool_calls: int  # Web 検索の回数
    function_calls: int = 0  # draft-db のツールの呼び出し回数（条件 D）
    max_request_input_tokens: int = 0  # 1回のリクエストの入力トークンの最大（長いコンテキストの割増の判定）
    api_retries: int  # API の例外による再試行の回数
    status: str  # ok / invalid_output / error
    error: str | None = None
    prediction_path: str | None = None
    research_path: str | None = None  # 条件 C・D の調査メモ
    draft_path: str | None = None  # 条件 D の自己検証前の予測


def _retry_prompt(user: str, error: str, text: str) -> str:
    return (
        f"{user}\n\n---\n前回の出力はスキーマに違反していました。\n{error}\n\n"
        f"前回の出力:\n{text[:20000]}\n\n修正した JSON だけを出力してください。"
    )


def _schema_error(res: RunnerResult, e: ValidationError) -> str:
    return f"stop_reason={res.stop_reason}: {e.error_count()}件のスキーマ違反: {str(e)[:2000]}"


def run_once(
    spec: ModelSpec,
    condition: str,
    run_no: int,
    prompt: Prompt,
    runner: RunnerFn,
    predictions_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    tools: ToolBox | None = None,
) -> RunLog:
    """1回の実行。段階は 調査（C・D）→ 整形 → 自己検証（D）。

    JSON を出力する段階（整形・自己検証）でスキーマに違反したら、ツールなしで1回だけ出し直す（attempts に数える）。
    """
    if prompt.uses_db and tools is None:
        raise ValueError(f"条件{condition}には draft-db のツールが必要です")
    started = datetime.now(UTC)
    run_id = f"{condition}-{spec.model_id}-{run_no:02d}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    t0 = time.monotonic()
    totals = {"in": 0, "out": 0, "reasoning": None, "tools": 0, "functions": 0, "max_in": 0, "retries": 0, "attempts": 0}
    status, error, path, research_path, draft_path = "invalid_output", None, None, None, None
    out_dir = predictions_dir / condition / spec.model_id

    def call(system: str, user: str, stage: Stage | None = None) -> RunnerResult:
        res, retries = call_with_retry(
            runner,
            (spec.model_id, system, user, MAX_OUTPUT_TOKENS),
            sleep,
            web_search=bool(stage and stage.web_search),
            tools=tools if stage and stage.use_db else None,
        )
        totals["retries"] += retries
        totals["in"] += res.input_tokens
        totals["out"] += res.output_tokens
        totals["tools"] += res.tool_calls
        totals["functions"] += res.function_calls
        totals["max_in"] = max(totals["max_in"], res.max_request_input_tokens)
        if res.reasoning_tokens is not None:
            totals["reasoning"] = (totals["reasoning"] or 0) + res.reasoning_tokens
        return res

    def json_stage(system: str, user: str, stage: Stage | None = None) -> tuple[PredictionOutput | None, str | None]:
        """JSON を出力させる。違反なら、エラーを添えてツールなしで1回だけ出し直す。"""
        res = call(system, user, stage)
        for retry in (False, True):
            totals["attempts"] += 1
            try:
                return PredictionOutput.model_validate_json(extract_json(res.text)), None
            except ValidationError as e:
                err = _schema_error(res, e)
                if retry:
                    return None, err
                res = call(prompt.system, _retry_prompt(user, err, res.text))
        raise AssertionError("unreachable")

    def save(name: str, text: str) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / name
        f.write_text(text, encoding="utf-8")
        return str(f)

    try:
        user, notes = prompt.user, ""
        if prompt.research:
            # 調査: Web 検索（と draft-db のツール）を使って、自由文の調査メモを書かせる
            res = call(prompt.research.system, prompt.research.user, prompt.research)
            notes = res.text
            if not notes.strip():
                raise ValueError(f"調査メモが空です（stop_reason={res.stop_reason}）")
            research_path = save(f"{run_id}.research.md", notes)
            user = prompt.user.replace(NOTES_PLACEHOLDER, notes)
        # 整形（条件 A・B はこれだけ）: ツールなしで JSON を出力させる
        output, error = json_stage(prompt.system, user)
        if output is not None and prompt.verify:
            # 自己検証: draft-db のツールで全選手を確かめさせ、直した JSON を出力させる
            draft_path = save(f"{run_id}.draft.json", output.model_dump_json(indent=1))
            verify_user = prompt.verify.user.replace(NOTES_PLACEHOLDER, notes).replace(
                DRAFT_PLACEHOLDER, output.model_dump_json(indent=1)
            )
            output, error = json_stage(prompt.verify.system, verify_user, prompt.verify)
        if output is not None:
            pred = Prediction(run_id=run_id, model_id=spec.model_id, condition=condition, output=output)
            path = save(f"{run_id}.json", pred.model_dump_json(indent=1))
            status, error = "ok", None
    except RetriesExhausted as e:  # API エラーも記録して次の実行へ進む
        totals["retries"] += e.retries
        status, error = "error", str(e)[:2000]
    except Exception as e:
        status, error = "error", f"{type(e).__name__}: {str(e)[:2000]}"

    return RunLog(
        run_id=run_id,
        model_id=spec.model_id,
        provider=spec.provider.value,
        condition=condition,
        run_no=run_no,
        prompt_sha256=prompt.sha256,
        started_at=started.isoformat(),
        duration_sec=round(time.monotonic() - t0, 2),
        attempts=totals["attempts"],
        input_tokens=totals["in"],
        output_tokens=totals["out"],
        reasoning_tokens=totals["reasoning"],
        cost_usd=round(spec.cost_usd(totals["in"], totals["out"], totals["tools"]), 6),
        tool_calls=totals["tools"],
        function_calls=totals["functions"],
        max_request_input_tokens=totals["max_in"],
        api_retries=totals["retries"],
        status=status,
        error=error,
        prediction_path=path,
        research_path=research_path,
        draft_path=draft_path,
    )


def get_runner(provider: Provider) -> RunnerFn:
    if provider == Provider.ANTHROPIC:
        from src.runners import anthropic_api

        return anthropic_api.call
    if provider == Provider.OPENAI:
        from src.runners import openai_api

        return openai_api.call
    from src.runners import gemini_api

    return gemini_api.call


def main(argv: list[str] | None = None, runner: RunnerFn | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--start", type=int, default=1, help="最初の実行番号")
    ap.add_argument("--budget", type=float, help="予算上限（USD）。省略時は環境変数 BUDGET_USD")
    ap.add_argument("--dry-run", action="store_true", help="見積もりだけ表示して終了")
    ap.add_argument("--predictions", type=Path, default=Path("predictions"))
    ap.add_argument("--runs", type=Path, default=Path("runs/runs.jsonl"))
    args = ap.parse_args(argv)

    spec = MODELS[args.model]
    prompt = build_prompt(args.condition)
    estimate = estimate_max_cost(spec, prompt, args.n)
    print(
        f"{spec.model_id} 条件{args.condition} × {args.n}回: 見積もり上限 ${estimate:.2f}"
        f"（実行日 {format_date(today_jst())}、prompt {prompt.sha256[:12]}）"
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

    call = runner or get_runner(spec.provider)
    tools = DraftDB.load() if prompt.uses_db else None
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for run_no in range(args.start, args.start + args.n):
        log = run_once(spec, args.condition, run_no, prompt, call, args.predictions, tools=tools)
        with args.runs.open("a", encoding="utf-8") as f:
            f.write(log.model_dump_json() + "\n")
        failed += log.status != "ok"
        print(
            f"  #{run_no} {log.status} attempts={log.attempts} tokens={log.input_tokens}/{log.output_tokens} "
            f"検索={log.tool_calls} ツール={log.function_calls} ${log.cost_usd:.4f} {log.duration_sec}s"
        )
        if log.error:
            print(f"    {log.error[:300]}", file=sys.stderr)
        if spec.long_context_input_tokens and log.max_request_input_tokens > spec.long_context_input_tokens:
            # cost_usd は通常の単価で計算しているので、実際の請求より低い
            print(
                f"    注意: 入力 {log.max_request_input_tokens} トークンのリクエストがあり、長いコンテキストの単価"
                f"（{spec.long_context_input_tokens} 超）が適用されます。記録したコストは過小です",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
