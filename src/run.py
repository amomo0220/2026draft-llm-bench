"""予測を実行して predictions/ に保存し、実行ログを runs/runs.jsonl に記録する。

現在の対応: 条件 A（情報なし）・B（候補者一覧をコンテキストに入れる）。C・D・H は未実装。

出力は各社の JSON モードで受け取り、PredictionOutput（src/schema.py）で検証する。
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
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.candidates import load_candidates
from src.models import MODELS, ModelSpec, Provider
from src.runners.base import RunnerResult
from src.schema import Prediction, PredictionOutput

PROMPTS = Path("prompts")
CONDITIONS = ("A", "B")
MAX_OUTPUT_TOKENS = 32_000
MAX_ATTEMPTS = 2  # 初回＋スキーマ違反時の出し直し1回
CHARS_PER_TOKEN = 1.0  # 日本語の見積もり用（実測で見直す）
API_RETRY_DELAYS = (10.0, 30.0, 90.0)  # 混雑（503）やレート制限など、API の例外時に待つ秒数

RunnerFn = Callable[[str, str, str, int], RunnerResult]


def call_with_retry(
    runner: RunnerFn, args: tuple[str, str, str, int], sleep: Callable[[float], None] = time.sleep
) -> tuple[RunnerResult, int]:
    """API の例外は待ってから再試行する。(結果, 再試行回数) を返し、再試行し尽くしたら最後の例外を投げる。"""
    for retries, delay in enumerate((*API_RETRY_DELAYS, None)):
        try:
            return runner(*args), retries
        except Exception:
            if delay is None:
                raise
            sleep(delay)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.system}\n\n{self.user}".encode()).hexdigest()


def _candidates_table(path: Path) -> str:
    rows = load_candidates(path)
    return "\n".join(f"{c.category.value},{c.name},{c.team},{c.position or ''}" for c in rows)


def build_prompt(condition: str, prompts_dir: Path = PROMPTS, candidates: Path = Path("data/candidates.csv")) -> Prompt:
    if condition not in CONDITIONS:
        raise ValueError(f"未対応の条件です: {condition}（対応: {', '.join(CONDITIONS)}）")
    schema = json.dumps(PredictionOutput.model_json_schema(), ensure_ascii=False, indent=1)
    user = (prompts_dir / "task.md").read_text(encoding="utf-8").replace("{{SCHEMA}}", schema)
    if condition == "B":
        context = (prompts_dir / "context_B.md").read_text(encoding="utf-8")
        user = context.replace("{{CANDIDATES}}", _candidates_table(candidates)) + "\n\n" + user
    return Prompt(system=(prompts_dir / "system.md").read_text(encoding="utf-8"), user=user)


def estimate_max_cost(spec: ModelSpec, prompt: Prompt, n: int) -> float:
    """見積もりの上限。出力は max_output_tokens を使い切り、全実行で出し直しが起きた場合。"""
    input_tokens = int((len(prompt.system) + len(prompt.user)) / CHARS_PER_TOKEN)
    per_attempt = spec.cost_usd(input_tokens + MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
    return per_attempt * MAX_ATTEMPTS * n


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
    tool_calls: int
    api_retries: int  # API の例外による再試行の回数
    status: str  # ok / invalid_output / error
    error: str | None = None
    prediction_path: str | None = None


def run_once(
    spec: ModelSpec,
    condition: str,
    run_no: int,
    prompt: Prompt,
    runner: RunnerFn,
    predictions_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> RunLog:
    started = datetime.now(UTC)
    run_id = f"{condition}-{spec.model_id}-{run_no:02d}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    t0 = time.monotonic()
    totals = {"in": 0, "out": 0, "reasoning": None, "tools": 0, "retries": 0}
    user, attempts, status, error, path = prompt.user, 0, "invalid_output", None, None

    try:
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            res, retries = call_with_retry(runner, (spec.model_id, prompt.system, user, MAX_OUTPUT_TOKENS), sleep)
            totals["retries"] += retries
            totals["in"] += res.input_tokens
            totals["out"] += res.output_tokens
            totals["tools"] += res.tool_calls
            if res.reasoning_tokens is not None:
                totals["reasoning"] = (totals["reasoning"] or 0) + res.reasoning_tokens
            try:
                output = PredictionOutput.model_validate_json(extract_json(res.text))
            except ValidationError as e:
                error = f"stop_reason={res.stop_reason}: {e.error_count()}件のスキーマ違反: {str(e)[:2000]}"
                user = (
                    f"{prompt.user}\n\n---\n前回の出力はスキーマに違反していました。\n{error}\n\n"
                    f"前回の出力:\n{res.text[:20000]}\n\n修正した JSON だけを出力してください。"
                )
                continue
            pred = Prediction(run_id=run_id, model_id=spec.model_id, condition=condition, output=output)
            out = predictions_dir / condition / spec.model_id / f"{run_id}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(pred.model_dump_json(indent=1), encoding="utf-8")
            status, error, path = "ok", None, str(out)
            break
    except Exception as e:  # API エラーも記録して次の実行へ進む
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
        attempts=attempts,
        input_tokens=totals["in"],
        output_tokens=totals["out"],
        reasoning_tokens=totals["reasoning"],
        cost_usd=round(spec.cost_usd(totals["in"], totals["out"]), 6),
        tool_calls=totals["tools"],
        api_retries=totals["retries"],
        status=status,
        error=error,
        prediction_path=path,
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
    print(f"{spec.model_id} 条件{args.condition} × {args.n}回: 見積もり上限 ${estimate:.2f}（prompt {prompt.sha256[:12]}）")
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
    args.runs.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    for run_no in range(args.start, args.start + args.n):
        log = run_once(spec, args.condition, run_no, prompt, call, args.predictions)
        with args.runs.open("a", encoding="utf-8") as f:
            f.write(log.model_dump_json() + "\n")
        failed += log.status != "ok"
        print(f"  #{run_no} {log.status} attempts={log.attempts} tokens={log.input_tokens}/{log.output_tokens} ${log.cost_usd:.4f} {log.duration_sec}s")
        if log.error:
            print(f"    {log.error[:300]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
