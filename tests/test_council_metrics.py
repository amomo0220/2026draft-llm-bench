from pathlib import Path

import pytest

from src.actual import ActualDraft
from src.candidates import load_candidates
from src.council import main as council_main
from src.council_metrics import (
    agreement,
    change_correctness,
    load_turns,
    main,
    name_corrections,
    observe,
    opinion_change_rate,
    opinion_changes,
    order_sensitivity,
    unique_candidates,
)
from src.matching import Master
from src.schema import TeamCode
from tests.council_fixtures import output
from tests.dummy_draft import actual_payload
from tests.test_council import FakeRunner

ACTUAL = ActualDraft.model_validate(actual_payload())  # G・T は 競合 一郎、M は ロッテ 一位、他は 選手X に入札


@pytest.fixture(scope="module")
def master() -> Master:
    return Master(load_candidates(Path("data/candidates.csv")))


def test_opinion_change_rate() -> None:
    r0 = {"teams": output({"G": "甲"}), "players": output()}
    r1 = {"teams": output({"G": "乙", "T": "乙"}), "players": output()}
    assert opinion_changes(r0, r1) == [("teams", TeamCode.G), ("teams", TeamCode.T)]
    assert opinion_change_rate(r0, r1) == pytest.approx(2 / 24)
    assert opinion_change_rate({}, {}) == 0.0


def test_change_correctness() -> None:
    r0 = {"teams": output({"G": "甲", "T": "競合 一郎", "C": "甲"})}
    r1 = {"teams": output({"G": "競合 一郎", "T": "甲", "C": "乙"})}
    # G: 正解に変えた（改善）、T: 正解から離れた（悪化）、C: どちらも外れ
    assert change_correctness(r0, r1, ACTUAL) == {"improved": 1, "worsened": 1, "neutral": 1}


def test_agreement_and_order_sensitivity() -> None:
    a, b = output(), output({"G": "甲", "T": "甲", "C": "甲"})
    assert agreement(a, a) == 1.0 and agreement(a, b) == pytest.approx(9 / 12)
    assert order_sensitivity([a, a, b]) == pytest.approx(9 / 12)
    assert order_sensitivity([a]) == 1.0


def test_unique_candidates() -> None:
    # 1つの予測は 12球団 × 2人（最有力 11人 ＋ 次点 12人。G と T の 競合 一郎 は同じ選手）
    assert unique_candidates([output()]) == 23
    assert unique_candidates([output(), output({"G": "甲"})]) == 24


def test_name_corrections(master: Master) -> None:
    # ダミーの名前はどれもマスタにない。最終予測から消えた名前（甲・乙）だけを数える
    r0 = [output({"G": "甲"}), output({"G": "乙"})]
    assert name_corrections(r0, output(), master) == 2
    assert name_corrections(r0, output({"G": "甲"}), master) == 1


def test_observe_from_council_log(tmp_path: Path, master: Master) -> None:
    log = tmp_path / "council.jsonl"
    argv = ["--predictions", str(tmp_path / "p"), "--runs", str(tmp_path / "r.jsonl"), "--council-log", str(log), "--budget", "1000"]
    assert council_main(argv, runner=FakeRunner(), sleep=lambda _: None) == 0
    report = observe(load_turns(log), master, ACTUAL)
    m_runs = [r for r in report["runs"] if r["condition"] == "M"]
    s_runs = [r for r in report["runs"] if r["condition"] == "S"]
    assert [r["run_no"] for r in m_runs] == [1, 2, 3] and len(s_runs) == 3
    # 偽の API は役割ごとに G の最有力候補だけを変える: Round 0 → 1 で各役の G が変わる
    assert m_runs[0]["opinion_change_rate"] == pytest.approx(3 / 36, abs=1e-4)
    assert m_runs[0]["change_correctness"] == {"improved": 0, "worsened": 0, "neutral": 3}  # G はどの案も外れ
    assert [r["first_speaker"] for r in m_runs] == ["teams", "players", "verifier"]
    assert m_runs[0]["first_speaker_agreement"] == pytest.approx(11 / 12, abs=1e-4)
    assert report["m_order_sensitivity"] == 1.0  # 偽の API の議長はいつも同じ予測
    assert all(r["k"] >= 1 and "トークン" in r["budget"] for r in s_runs)


def test_main_writes_report(tmp_path: Path) -> None:
    log = tmp_path / "council.jsonl"
    argv = ["--predictions", str(tmp_path / "p"), "--runs", str(tmp_path / "r.jsonl"), "--council-log", str(log), "--budget", "1000", "--run-nos", "1"]
    council_main(argv, runner=FakeRunner(), sleep=lambda _: None)
    out = tmp_path / "obs.json"
    assert main(["--council-log", str(log), "--out", str(out)]) == 0 and out.exists()
    assert main(["--council-log", str(tmp_path / "none.jsonl")]) == 1
