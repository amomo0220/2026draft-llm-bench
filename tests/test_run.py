import json
from pathlib import Path

import pytest

from src.models import MODELS
from src.run import RunLog, build_prompt, estimate_max_cost, extract_json, main, run_once
from src.runners.base import RunnerResult
from src.schema import Prediction
from tests.dummy_draft import prediction_payload

SPEC = MODELS["gemini-3.8-flash"]
VALID = json.dumps(prediction_payload()["output"], ensure_ascii=False)


def _runner(*texts: str):
    """呼ばれるたびに texts を順に返す偽の API。"""
    calls: list[str] = []
    searches: list[bool] = []

    def call(model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False) -> RunnerResult:
        calls.append(user)
        searches.append(web_search)
        return RunnerResult(
            text=texts[len(calls) - 1], input_tokens=1000, output_tokens=500, reasoning_tokens=100, tool_calls=3 if web_search else 0
        )

    call.calls = calls  # type: ignore[attr-defined]
    call.searches = searches  # type: ignore[attr-defined]
    return call


# --- プロンプト ---


def test_build_prompt_b_includes_candidates_and_schema() -> None:
    p = build_prompt("B")
    assert "織田 翔希" in p.user and "藤澤 涼介" in p.user
    assert '"first_round"' in p.user  # スキーマが埋め込まれている
    assert "{{" not in p.user


def test_build_prompt_a_has_no_candidates() -> None:
    p = build_prompt("A")
    assert "織田 翔希" not in p.user
    assert build_prompt("A").sha256 == p.sha256  # 同じ入力なら同じハッシュ
    assert build_prompt("B").sha256 != p.sha256


def test_build_prompt_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError):
        build_prompt("D")


def test_estimate_scales_with_n() -> None:
    p = build_prompt("B")
    assert estimate_max_cost(SPEC, p, 3) == pytest.approx(3 * estimate_max_cost(SPEC, p, 1))


@pytest.mark.parametrize(
    "text",
    [VALID, f"```json\n{VALID}\n```", f"以下が予測です。\n{VALID}\n以上です。"],
)
def test_extract_json(text: str) -> None:
    assert json.loads(extract_json(text)) == json.loads(VALID)


# --- 1回の実行 ---


def test_run_once_ok(tmp_path: Path) -> None:
    log = run_once(SPEC, "B", 1, build_prompt("B"), _runner(VALID), tmp_path)
    assert log.status == "ok" and log.attempts == 1
    assert log.input_tokens == 1000 and log.output_tokens == 500 and log.reasoning_tokens == 100
    assert log.cost_usd == pytest.approx(SPEC.cost_usd(1000, 500), abs=1e-6)
    pred = Prediction.model_validate_json(Path(log.prediction_path or "").read_text(encoding="utf-8"))
    assert pred.condition == "B" and pred.model_id == SPEC.model_id and pred.run_id == log.run_id
    assert Path(log.prediction_path or "").parent == tmp_path / "B" / SPEC.model_id


def test_run_once_retries_once_with_error(tmp_path: Path) -> None:
    runner = _runner('{"first_round": []}', VALID)
    log = run_once(SPEC, "B", 1, build_prompt("B"), runner, tmp_path)
    assert log.status == "ok" and log.attempts == 2
    assert log.input_tokens == 2000  # 2回分を合算
    assert "スキーマに違反" in runner.calls[1]


def test_run_once_gives_up_after_two_invalid(tmp_path: Path) -> None:
    log = run_once(SPEC, "B", 1, build_prompt("B"), _runner("だめ", "まだだめ"), tmp_path)
    assert log.status == "invalid_output" and log.attempts == 2
    assert log.prediction_path is None and log.error
    assert not list(tmp_path.rglob("*.json"))


def test_run_once_records_api_error_after_retries(tmp_path: Path) -> None:
    calls: list[int] = []
    waits: list[float] = []

    def boom(*_: object, **__: object) -> RunnerResult:
        calls.append(1)
        raise RuntimeError("503 UNAVAILABLE")

    log = run_once(SPEC, "B", 1, build_prompt("B"), boom, tmp_path, sleep=waits.append)
    assert log.status == "error" and "503" in (log.error or "")
    assert len(calls) == 4 and waits == [10.0, 30.0, 90.0]
    assert log.api_retries == 3


def test_run_once_recovers_from_transient_error(tmp_path: Path) -> None:
    state = {"n": 0}

    def flaky(*_: object, **__: object) -> RunnerResult:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("503 UNAVAILABLE")
        return RunnerResult(text=VALID, input_tokens=1000, output_tokens=500)

    log = run_once(SPEC, "B", 1, build_prompt("B"), flaky, tmp_path, sleep=lambda _: None)
    assert log.status == "ok" and log.api_retries == 1 and log.attempts == 1


# --- CLI ---


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return ["--condition", "B", "--model", SPEC.model_id, "--predictions", str(tmp_path / "p"), "--runs", str(tmp_path / "runs.jsonl"), *extra]


def test_main_requires_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUDGET_USD", raising=False)
    assert main(_args(tmp_path), runner=_runner(VALID)) == 1
    assert not (tmp_path / "runs.jsonl").exists()


def test_main_stops_over_budget(tmp_path: Path) -> None:
    assert main(_args(tmp_path, "--budget", "0.0001"), runner=_runner(VALID)) == 1
    assert not (tmp_path / "runs.jsonl").exists()


def test_main_dry_run_does_not_call(tmp_path: Path) -> None:
    runner = _runner()
    assert main(_args(tmp_path, "--dry-run"), runner=runner) == 0
    assert runner.calls == []


def test_main_runs_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUDGET_USD", "100")
    assert main(_args(tmp_path, "--n", "2", "--start", "3"), runner=_runner(VALID, VALID)) == 0
    logs = [RunLog.model_validate_json(line) for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [log.run_no for log in logs] == [3, 4]
    assert all(log.status == "ok" for log in logs)
    assert len(list((tmp_path / "p").rglob("*.json"))) == 2


# --- 条件 C（Web 検索） ---


def test_build_prompt_c_asks_to_search_without_candidates() -> None:
    p = build_prompt("C")
    assert "Web 検索" in p.user
    assert "織田 翔希" not in p.user


def test_estimate_c_includes_search() -> None:
    p = build_prompt("C")
    assert estimate_max_cost(SPEC, p, 1, web_search=True) > estimate_max_cost(SPEC, p, 1)


def test_run_once_c_searches_only_on_first_attempt(tmp_path: Path) -> None:
    runner = _runner('{"first_round": []}', VALID)
    log = run_once(SPEC, "C", 1, build_prompt("C"), runner, tmp_path)
    assert runner.searches == [True, False]  # 出し直しでは検索させない
    assert log.status == "ok" and log.tool_calls == 3
    assert log.cost_usd == pytest.approx(SPEC.cost_usd(2000, 1000, 3), abs=1e-6)


def test_run_once_b_never_searches(tmp_path: Path) -> None:
    runner = _runner(VALID)
    run_once(SPEC, "B", 1, build_prompt("B"), runner, tmp_path)
    assert runner.searches == [False]
