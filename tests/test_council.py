import json
from pathlib import Path

import pytest

from src.council import (
    CHAIR,
    MODEL_ID,
    ORDERS,
    Caller,
    CouncilTurn,
    TurnLog,
    build_council_prompts,
    estimate_cost,
    main,
    run_council,
    run_single,
    single_should_continue,
)
from src.models import MODELS
from src.run import build_prompt
from src.runners.base import RunnerResult
from src.scorer import load_predictions
from src.schema import PredictionOutput
from tests.council_fixtures import output, turn_json

SPEC = MODELS[MODEL_ID]
ROLE_OF = {"球団動向担当": "teams", "選手評価担当": "players", "検証担当": "verifier", "議長": CHAIR}


def _role(system: str) -> str:
    return next(role for label, role in ROLE_OF.items() if f"あなたの役割: {label}" in system)


def _round(user: str) -> str:
    for key, name in (("今回の発言: 独立提案", "round0"), ("今回の発言: 反論と修正", "round1"), ("今回の発言: 最終予測", "final")):
        if key in user:
            return name
    return "single"


class FakeRunner:
    """役割と発言の種類ごとに決まった予測を返す偽の API。bad に入れた (役割, 発言) は不正な JSON を返す。"""

    def __init__(self, tokens: tuple[int, int] = (1000, 500), bad: dict[tuple[str, str], int] | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []  # (役割, 発言, user)
        self.tokens = tokens
        self.bad = dict(bad or {})

    def __call__(self, model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False) -> RunnerResult:
        rnd = _round(user)
        role = "single" if rnd == "single" else _role(system)
        self.calls.append((role, rnd, user))
        if self.bad.get((role, rnd), 0) > 0:
            self.bad[(role, rnd)] -= 1
            text = "{壊れた JSON"
        elif rnd == "single":
            text = output({"G": f"S案{len(self.calls)}"}).model_dump_json()
        else:
            text = turn_json(output({"G": f"{role}-{rnd}"}), argument=f"{role} の {rnd} の発言")
        return RunnerResult(text=text, input_tokens=self.tokens[0], output_tokens=self.tokens[1])


@pytest.fixture(scope="module")
def prompts():
    return build_council_prompts()


def _caller(runner: FakeRunner) -> Caller:
    return Caller(SPEC, runner, sleep=lambda _: None)


# --- プロンプト ---


def test_prompts_have_roles_and_constraints(prompts) -> None:
    for role, label in (("teams", "球団動向担当"), ("players", "選手評価担当"), ("verifier", "検証担当"), (CHAIR, "議長")):
        system = prompts.system(role)
        assert f"あなたの役割: {label}" in system
        assert "選手個人の短所や欠点を述べない" in system  # 未成年の選手への配慮（COUNCIL.md 5章）
        assert "特定の球団の編成会議ではありません" in system  # 中立の立場（COUNCIL.md 3.1）
    user = prompts.user("round1", "ここに議論")
    assert "ここに議論" in user and "織田 翔希" in user and '"argument"' in user and "{{" not in user


def test_estimate_upper_bound_exceeds_typical(prompts) -> None:
    upper = sum(estimate_cost(SPEC, prompts, 3))
    typical = sum(estimate_cost(SPEC, prompts, 3, 6000, attempts=1))
    assert upper > typical > 0
    assert sum(estimate_cost(SPEC, prompts, 1)) == pytest.approx(upper / 3)


# --- M ---


@pytest.mark.parametrize("run_no", sorted(ORDERS))
def test_council_speaking_order(prompts, run_no: int) -> None:
    runner = FakeRunner()
    out = run_council(_caller(runner), prompts, run_no)
    assert out.error is None and out.output is not None
    assert [(r, n) for r, n, _ in runner.calls] == [
        ("teams", "round0"),
        ("players", "round0"),
        ("verifier", "round0"),
        *[(r, "round1") for r in ORDERS[run_no]],
        (CHAIR, "final"),
    ]
    # 議長の予測が最終予測。Round 1 の案は発言順に並ぶ（M-avg の材料）
    assert out.output == output({"G": "chair-final"})
    assert out.round1 == [output({"G": f"{r}-round1"}) for r in ORDERS[run_no]]
    assert [t.speaker_index for t in out.turns if t.round == "round1"] == [0, 1, 2]
    assert out.tokens == 7 * 1500


def test_round1_sees_earlier_speakers_only(prompts) -> None:
    runner = FakeRunner()
    run_council(_caller(runner), prompts, 1)
    users = {(r, n): u for r, n, u in runner.calls}
    assert "teams の round0 の発言" in users[("teams", "round1")]  # 全員の Round 0 を読む
    assert "verifier の round0 の発言" in users[("teams", "round1")]
    assert "teams の round1 の発言" not in users[("teams", "round1")]
    assert "teams の round1 の発言" in users[("players", "round1")]
    assert "players の round1 の発言" not in users[("players", "round1")]
    assert "players の round1 の発言" in users[("verifier", "round1")]
    final = users[(CHAIR, "final")]
    assert all(f"{r} の round1 の発言" in final for r in ORDERS[1]) and "round0 の発言" not in final


def test_council_retries_invalid_output_once(prompts) -> None:
    runner = FakeRunner(bad={("players", "round0"): 1})
    out = run_council(_caller(runner), prompts, 1)
    assert out.output is not None
    players = next(t for t in out.turns if t.role == "players" and t.round == "round0")
    assert players.attempts == 2 and players.status == "ok" and players.input_tokens == 2000
    assert "前回の出力はスキーマに違反していました" in runner.calls[2][2]


def test_council_stops_when_a_turn_fails(prompts) -> None:
    runner = FakeRunner(bad={("players", "round0"): 2})
    out = run_council(_caller(runner), prompts, 1)
    assert out.output is None and "選手評価担当（round0）" in (out.error or "")
    assert len(runner.calls) == 3  # teams 1回＋players 2回で打ち切る
    assert out.turns[-1].status == "invalid_output" and out.turns[-1].content == "{壊れた JSON"


def test_turn_log_records_argument_and_prediction(prompts) -> None:
    out = run_council(_caller(FakeRunner()), prompts, 1)
    t = out.turns[0]
    assert t.content == "teams の round0 の発言"
    assert PredictionOutput.model_validate(t.prediction_json) == output({"G": "teams-round0"})
    assert t.prompt_sha256 and t.model_id == MODEL_ID and t.cost_usd == pytest.approx(SPEC.cost_usd(1000, 500))
    assert CouncilTurn.model_validate_json(turn_json(output())).argument == "ダミーの発言"


# --- S ---


@pytest.mark.parametrize(
    ("used", "per_call", "target", "expected"),
    [
        (0, [], 10_000, True),  # 1回目は必ず呼ぶ
        (8_000, [4_000, 4_000], 10_000, False),  # 次の1回で 12,000（+20%）になるので止める
        (9_000, [1_500] * 6, 10_000, False),  # 下限（-10%）に届いた
        (7_500, [1_500] * 5, 10_000, True),
    ],
)
def test_single_should_continue(used: int, per_call: list[int], target: int, expected: bool) -> None:
    assert single_should_continue(used, per_call, target) is expected


def test_single_matches_budget() -> None:
    runner = FakeRunner(tokens=(1000, 500))
    out = run_single(_caller(runner), build_prompt("B"), 1, target_tokens=10_000)
    summary = out.turns[-1]
    assert len(runner.calls) == 6 and summary.k == 6 and summary.status == "ok" and summary.target_tokens == 10_000
    assert out.tokens == 9_000 and out.output is not None
    assert runner.calls[0][2] == build_prompt("B").user  # 条件B と同じプロンプト


def test_single_reports_budget_mismatch() -> None:
    out = run_single(_caller(FakeRunner(tokens=(3000, 1000))), build_prompt("B"), 1, target_tokens=10_000)
    assert out.turns[-1].status == "budget_mismatch" and out.turns[-1].k == 2


def test_single_skips_invalid_samples() -> None:
    runner = FakeRunner(bad={("single", "single"): 2})  # 1回目は出し直しても不正
    out = run_single(_caller(runner), build_prompt("B"), 1, target_tokens=6_000)
    samples = [t for t in out.turns if t.round == "sample"]
    assert samples[0].status == "invalid_output" and out.turns[-1].k == len(samples) - 1


# --- まとめて実行 ---


def test_main_saves_final_predictions_only(tmp_path: Path) -> None:
    preds, runs, council = tmp_path / "predictions", tmp_path / "runs.jsonl", tmp_path / "council.jsonl"
    argv = ["--predictions", str(preds), "--runs", str(runs), "--council-log", str(council), "--budget", "1000", "--run-nos", "1,3"]
    assert main(argv, runner=FakeRunner(), sleep=lambda _: None) == 0

    loaded = load_predictions(preds, include_late=False)  # 採点スクリプトがそのまま読める
    assert sorted(p.condition for p in loaded) == ["M", "M", "M-avg", "M-avg", "S", "S"]
    assert {p.model_id for p in loaded} == {MODEL_ID}
    logs = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    assert [(r["condition"], r["run_no"]) for r in logs] == [("M", 1), ("M-avg", 1), ("S", 1), ("M", 3), ("M-avg", 3), ("S", 3)]
    assert all(r["status"] == "ok" for r in logs)
    turns = [TurnLog.model_validate_json(line) for line in council.read_text(encoding="utf-8").splitlines()]
    assert sum(t.condition == "M" for t in turns) == 14 and sum(t.round == "summary" for t in turns) == 2


def test_main_stops_over_budget(tmp_path: Path) -> None:
    runner = FakeRunner()
    assert main(["--predictions", str(tmp_path), "--budget", "1"], runner=runner) == 1
    assert runner.calls == []


def test_main_requires_predictions_dir() -> None:
    with pytest.raises(SystemExit):
        main(["--dry-run"])
