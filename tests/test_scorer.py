import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.actual import ActualDraft
from src.candidates import COLUMNS, Candidate, Category
from src.matching import Master
from src.schema import FirstRoundBid, Prediction, TeamCode
from src.scorer import NameCheck, Verdict, brier, main, pending_names, score_run, summarize, top_candidate
from tests.dummy_draft import actual_payload, player, prediction_payload

ACTUAL = ActualDraft.model_validate(actual_payload())
EMPTY_MASTER = Master([])


def _pred(payload: dict[str, Any] | None = None) -> Prediction:
    return Prediction.model_validate(payload or prediction_payload())


def _bid(cands: list[tuple[str, float]]) -> FirstRoundBid:
    return FirstRoundBid.model_validate(
        {"team": "G", "candidates": [{**player(n), "prob": p} for n, p in cands]}
    )


def _team_bid(payload: dict[str, Any], team: TeamCode) -> dict[str, Any]:
    return next(b for b in payload["output"]["first_round"] if b["team"] == team.value)


# --- ブライアースコア・最尤候補 ---


def test_brier_listed() -> None:
    assert brier(_bid([("A", 0.6), ("B", 0.4)]), "A") == pytest.approx(0.4**2 + 0.4**2)


def test_brier_unlisted_counts_other() -> None:
    assert brier(_bid([("A", 0.6), ("B", 0.4)]), "C") == pytest.approx(0.36 + 0.16 + 1.0)


def test_brier_perfect_and_normalized() -> None:
    assert brier(_bid([("A", 0.995)]), "A") == pytest.approx(0.0)
    # 合計 0.995 は 1.0 に正規化してから計算する
    p_a = 0.5 / 0.995
    assert brier(_bid([("A", 0.5), ("B", 0.495)]), "A") == pytest.approx((1 - p_a) ** 2 + (1 - p_a) ** 2)


def test_brier_matches_variant_names() -> None:
    assert brier(_bid([("𠮷井 優都己", 1.0)]), "吉井優都己") == pytest.approx(0.0)


def test_top_candidate_tie_takes_first() -> None:
    assert top_candidate(_bid([("A", 0.5), ("B", 0.5)])).player_name == "A"


# --- 1位入札 ---


def test_perfect_first_round() -> None:
    s = score_run(_pred(), ACTUAL, EMPTY_MASTER, {})
    assert s.first_round_hits == 12
    assert s.first_round_position_hits == 12
    assert s.first_round_brier == pytest.approx(0.32)


def test_first_round_miss_and_position_hit() -> None:
    payload = prediction_payload()
    cands = _team_bid(payload, TeamCode.H)["candidates"]
    cands[0]["prob"], cands[1]["prob"] = 0.3, 0.7  # 次点（投手）が最尤に
    s = score_run(_pred(payload), ACTUAL, EMPTY_MASTER, {})
    assert s.first_round_hits == 11
    assert s.first_round_position_hits == 12  # 選手は外れたがポジションは一致


def test_contested_team_scored_by_bid_not_final() -> None:
    # T は抽選で外れたが、採点対象は入札（競合 一郎）なので的中
    s = score_run(_pred(), ACTUAL, EMPTY_MASTER, {})
    assert s.first_round_hits == 12


# --- ロッテ ---


def test_lotte_rounds() -> None:
    s = score_run(_pred(), ACTUAL, EMPTY_MASTER, {})
    assert s.lotte_rounds_scored == 4  # 5位は実際の指名がないので対象外
    assert s.lotte_round_hits == 4
    assert s.lotte_position_hits == 4
    assert s.lotte_top5_hits == 4


def test_lotte_top5_uses_final_when_lottery_lost() -> None:
    payload = actual_payload()
    m = next(r for r in payload["first_round"] if r["team"] == "M")
    m["won_lottery"], m["final"] = False, player("ロッテ 外れ一位")
    actual = ActualDraft.model_validate(payload)

    s = score_run(_pred(), actual, EMPTY_MASTER, {})
    assert s.lotte_round_hits == 4  # 1位は入札で採点するので的中のまま
    assert s.lotte_top5_hits == 3  # 獲得したのは外れ1位なので、予測の1位は獲得選手に含まれない


def test_lotte_order_swapped() -> None:
    payload = prediction_payload()
    lotte = payload["output"]["lotte"]
    lotte[1]["round"], lotte[2]["round"] = 3, 2  # 二位と三位を入れ替え
    s = score_run(_pred(payload), ACTUAL, EMPTY_MASTER, {})
    assert s.lotte_round_hits == 2
    assert s.lotte_top5_hits == 4  # 順位は違っても獲得選手には含まれる


# --- ハルシネーション ---


def _cand(pid: str, name: str, team: str, position: str | None = None) -> Candidate:
    return Candidate(
        player_id=pid, name=name, category=Category.HIGH_SCHOOL, team=team, position=position, source_url="https://example.com/"
    )


def test_name_classification() -> None:
    payload = prediction_payload()
    cands = _team_bid(payload, TeamCode.G)["candidates"]
    cands[0].update(player("競合 一郎", "ダミー高校", "投手"))  # 所属表記ゆれは誤りにしない
    cands[1].update(player("所属 違い", "別の高", "投手"))
    _team_bid(payload, TeamCode.H)["candidates"][1].update(player("位置 違い", "ダミー高", "投手"))
    _team_bid(payload, TeamCode.C)["candidates"][1].update(player("指名 不可", "ダミー高", "投手"))
    _team_bid(payload, TeamCode.D)["candidates"][1].update(player("架空 太郎", "ダミー高", "投手"))

    master = Master(
        [
            _cand("HS-001", "競合 一郎", "ダミー"),
            _cand("HS-002", "所属 違い", "本当の高校"),
            _cand("HS-003", "位置 違い", "ダミー高", "遊撃手"),
        ]
    )
    checks = {
        "指名不可": NameCheck(name="指名 不可", verdict=Verdict.INELIGIBLE, checked_on="2026-10-23"),
        "架空太郎": NameCheck(name="架空 太郎", verdict=Verdict.NOT_FOUND, checked_on="2026-10-23"),
    }
    s = score_run(_pred(payload), ACTUAL, master, checks)
    assert s.names_matched == 3
    assert s.affiliation_errors == 1
    assert s.position_errors == 1
    assert s.names_ineligible == 1
    assert s.names_hallucinated == 1
    assert s.names_pending == s.names_total - 5


def test_names_are_deduplicated() -> None:
    # 次点G〜次点M（12人）、1位入札の11人（G と T が同じ選手）、ロッテ二〜五位（4人、一位は入札と重複）
    s = score_run(_pred(), ACTUAL, EMPTY_MASTER, {})
    assert s.names_total == 12 + 11 + 4


def test_pending_names_counts_runs() -> None:
    preds = [_pred(prediction_payload(run_id=f"r{i}")) for i in range(3)]
    master = Master([_cand("HS-001", "競合 一郎", "ダミー高")])
    rows = pending_names(preds, master, {})
    assert all(r["runs"] == 3 for r in rows)
    assert "競合 一郎" not in [r["name"] for r in rows]


# --- 集計・CLI ---


def test_summarize() -> None:
    a = score_run(_pred(), ACTUAL, EMPTY_MASTER, {})
    b = a.model_copy(update={"run_id": "run-2", "first_round_hits": 10})
    c = a.model_copy(update={"run_id": "run-3", "condition": "C"})
    rows = summarize([a, b, c])
    assert [(r["condition"], r["n_runs"]) for r in rows] == [("B", 2), ("C", 1)]
    assert rows[0]["first_round_hits_mean"] == 11.0
    assert rows[0]["first_round_hits_sd"] == pytest.approx(1.4142, abs=1e-4)
    assert rows[1]["first_round_hits_sd"] == ""


def test_cli_end_to_end(tmp_path: Path) -> None:
    actual = tmp_path / "actual.json"
    actual.write_text(json.dumps(actual_payload(), ensure_ascii=False), encoding="utf-8")
    preds = tmp_path / "predictions"
    (preds / "late").mkdir(parents=True)
    (preds / "run-1.json").write_text(json.dumps(prediction_payload("run-1"), ensure_ascii=False), encoding="utf-8")
    (preds / "late" / "run-late.json").write_text(
        json.dumps(prediction_payload("run-late", condition="B-late"), ensure_ascii=False), encoding="utf-8"
    )
    cands = tmp_path / "candidates.csv"
    cands.write_text(",".join(COLUMNS) + "\nHS-001,競合 一郎,,高校,ダミー高,,,https://example.com/\n", encoding="utf-8")
    checks = tmp_path / "name_checks.csv"
    checks.write_text("name,verdict,checked_on,source_url,note\n架空 太郎,not_found,2026-10-23,,\n", encoding="utf-8")
    out = tmp_path / "results"

    args = ["--actual", str(actual), "--predictions", str(preds), "--candidates", str(cands)]
    args += ["--name-checks", str(checks), "--out", str(out)]
    assert main(args) == 0
    lines = (out / "run_scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["run-1"]  # late は既定で除外
    with (out / "summary.csv").open(encoding="utf-8") as f:
        assert next(csv.DictReader(f))["first_round_hits_mean"] == "12.0"
    assert (out / "pending_names.csv").exists()

    assert main(args + ["--include-late"]) == 0
    assert len((out / "run_scores.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_cli_no_predictions(tmp_path: Path) -> None:
    actual = tmp_path / "actual.json"
    actual.write_text(json.dumps(actual_payload(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "predictions").mkdir()
    assert main(["--actual", str(actual), "--predictions", str(tmp_path / "predictions")]) == 1
