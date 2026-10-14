import json
from datetime import date, datetime
from pathlib import Path

import pytest

from src.models import MODELS
from src.run import DRAFT_PLACEHOLDER, NOTES_PLACEHOLDER, Prompt, RunLog, build_prompt, estimate_max_cost, extract_json, main, run_once
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
        build_prompt("H")


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


def test_build_prompt_c_has_research_stage_without_candidates() -> None:
    p = build_prompt("C")
    assert p.research is not None
    assert "Web 検索" in p.research.user
    assert "{{SCHEMA}}" not in p.research.user and "first_round" not in p.research.user  # 調査では出力形式を求めない
    assert NOTES_PLACEHOLDER in p.user and '"first_round"' in p.user
    assert "織田 翔希" not in p.user + p.research.user


def test_prompt_hash_covers_research_stage() -> None:
    p = build_prompt("C")
    assert p.sha256 != Prompt(p.system, p.user).sha256


def test_estimate_c_includes_research_and_search() -> None:
    p = build_prompt("C")
    assert estimate_max_cost(SPEC, p, 1) > estimate_max_cost(SPEC, Prompt(p.system, p.user), 1)


def test_run_once_c_researches_then_formats(tmp_path: Path) -> None:
    runner = _runner("調査メモ: 某球団が1位を公言 https://example.com/a", '{"first_round": []}', VALID)
    log = run_once(SPEC, "C", 1, build_prompt("C"), runner, tmp_path)
    assert runner.searches == [True, False, False]  # 検索は調査だけ。整形と出し直しでは検索させない
    assert "Web 検索" in runner.calls[0]
    assert "https://example.com/a" in runner.calls[1] and NOTES_PLACEHOLDER not in runner.calls[1]
    assert "https://example.com/a" in runner.calls[2]  # 出し直しにもメモを入れる
    assert log.status == "ok" and log.attempts == 2 and log.tool_calls == 3
    assert log.cost_usd == pytest.approx(SPEC.cost_usd(3000, 1500, 3), abs=1e-6)
    assert Path(log.research_path).read_text(encoding="utf-8").startswith("調査メモ")
    assert Path(log.research_path).name == f"{log.run_id}.research.md"


def test_run_once_c_empty_notes_is_error(tmp_path: Path) -> None:
    runner = _runner("  ", VALID)
    log = run_once(SPEC, "C", 1, build_prompt("C"), runner, tmp_path)
    assert log.status == "error" and "調査メモが空" in log.error
    assert len(runner.calls) == 1 and log.prediction_path is None


def test_run_once_b_never_searches(tmp_path: Path) -> None:
    runner = _runner(VALID)
    run_once(SPEC, "B", 1, build_prompt("B"), runner, tmp_path)
    assert runner.searches == [False]


# --- 条件 D（Web 検索＋draft-db＋自己検証） ---


class _Tools:
    specs: list = []

    def call(self, name: str, args: dict) -> str:
        return "{}"


def _tool_runner(*texts: str):
    """tools を受け取る偽の API。呼び出しごとに (web_search, tools があるか) を記録する。"""
    calls: list[tuple[str, str]] = []
    flags: list[tuple[bool, bool]] = []

    def call(model_id, system, user, max_output_tokens, web_search=False, tools=None) -> RunnerResult:
        calls.append((system, user))
        flags.append((web_search, tools is not None))
        return RunnerResult(
            text=texts[len(calls) - 1], input_tokens=1000, output_tokens=500, tool_calls=2 if web_search else 0,
            function_calls=4 if tools is not None else 0,
        )

    call.calls = calls  # type: ignore[attr-defined]
    call.flags = flags  # type: ignore[attr-defined]
    return call


def test_build_prompt_d_stages() -> None:
    p = build_prompt("D")
    assert p.research and p.research.web_search and p.research.use_db and "search_players" in p.research.user
    assert p.verify and not p.verify.web_search and p.verify.use_db
    assert NOTES_PLACEHOLDER in p.verify.user and DRAFT_PLACEHOLDER in p.verify.user and '"first_round"' in p.verify.user
    assert p.uses_db and not build_prompt("C").uses_db
    assert "織田 翔希" not in p.user + p.research.user + p.verify.user  # マスタは最初から渡さない


def test_estimate_d_exceeds_c() -> None:
    assert estimate_max_cost(SPEC, build_prompt("D"), 1) > estimate_max_cost(SPEC, build_prompt("C"), 1)


def test_run_once_d_research_format_verify(tmp_path: Path) -> None:
    draft = prediction_payload()["output"]
    draft["lotte"][0]["player_name"] = "確認前の選手"
    runner = _tool_runner("調査メモ https://example.com/a", json.dumps(draft, ensure_ascii=False), VALID)
    log = run_once(SPEC, "D", 1, build_prompt("D"), runner, tmp_path, tools=_Tools())

    assert runner.flags == [(True, True), (False, False), (False, True)]  # 調査・整形・自己検証
    verify_user = runner.calls[2][1]
    assert "https://example.com/a" in verify_user and "確認前の選手" in verify_user
    assert log.status == "ok" and log.attempts == 2 and log.tool_calls == 2 and log.function_calls == 8
    assert "確認前の選手" in Path(log.draft_path).read_text(encoding="utf-8")
    pred = Prediction.model_validate_json(Path(log.prediction_path).read_text(encoding="utf-8"))
    assert pred.output.lotte[0].player_name != "確認前の選手"  # 自己検証の出力を保存する
    assert log.cost_usd == pytest.approx(SPEC.cost_usd(3000, 1500, 2), abs=1e-6)


def test_run_once_d_verify_retry_has_no_tools(tmp_path: Path) -> None:
    runner = _tool_runner("メモ", VALID, "壊れた出力", VALID)
    log = run_once(SPEC, "D", 1, build_prompt("D"), runner, tmp_path, tools=_Tools())
    assert runner.flags[3] == (False, False) and "前回の出力:\n壊れた出力" in runner.calls[3][1]
    assert log.status == "ok" and log.attempts == 3


def test_run_once_d_invalid_verify_output_is_not_saved(tmp_path: Path) -> None:
    runner = _tool_runner("メモ", VALID, "壊れた出力", "まだ壊れている")
    log = run_once(SPEC, "D", 1, build_prompt("D"), runner, tmp_path, tools=_Tools())
    assert log.status == "invalid_output" and log.prediction_path is None and log.draft_path


def test_run_once_d_requires_tools(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_once(SPEC, "D", 1, build_prompt("D"), _tool_runner(), tmp_path)


# --- 実行日 ---


@pytest.mark.parametrize("condition", ["A", "B", "C", "D"])
def test_build_prompt_fills_today(condition: str) -> None:
    p = build_prompt(condition, today=date(2026, 10, 18))
    texts = [p.system, p.user] + [t for s in (p.research, p.verify) if s for t in (s.system, s.user)]
    assert all("{{TODAY}}" not in t for t in texts)
    assert "今日は 2026年10月18日 です。" in (p.research.user if p.research else p.user)


def test_prompt_hash_changes_with_date() -> None:
    assert build_prompt("C", today=date(2026, 10, 18)).sha256 != build_prompt("C", today=date(2026, 10, 19)).sha256


def test_today_jst_is_jst() -> None:
    from src.run import JST, today_jst

    assert today_jst() == datetime.now(JST).date()
