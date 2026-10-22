"""答え合わせの表（ダミーの正解データで確かめる）。"""

import copy

from src.actual import ActualDraft
from src.matching import Master
from src.report_results import (
    BASELINE_LABEL,
    first_round_result_table,
    lotte_result_table,
    sections,
    summary_table,
)
from src.schema import FirstRoundBid, Prediction
from src.scorer import score_run
from tests.dummy_draft import actual_payload, prediction_payload

ACTUAL = ActualDraft.model_validate(actual_payload())


def _pred(run_id: str, model_id: str = "m1", miss_g: bool = False) -> Prediction:
    payload = copy.deepcopy(prediction_payload(run_id, model_id, "D"))
    if miss_g:  # 巨人の本命を外す
        g = payload["output"]["first_round"][0]
        g["candidates"][0]["prob"], g["candidates"][1]["prob"] = 0.4, 0.6
    return Prediction.model_validate(payload)


BASE = [FirstRoundBid.model_validate({"team": "G", "candidates": [{"player_name": "別 人", "affiliation": "x", "position": "投手", "prob": 1.0}]})]


def test_first_round_result_marks_hits_and_misses() -> None:
    table = first_round_result_table({"m1": [_pred("r1"), _pred("r2", miss_g=True)]}, ACTUAL, BASE)
    assert table.header == ["球団", "実際の1位入札", "m1", BASELINE_LABEL]
    assert table.rows[0] == ["巨人", "競合 一郎・投手", "○ 1/2", "× 別 人"]
    assert table.rows[1][-1] == "—"  # メディア予想のない球団
    assert table.highlight[0] is True


def test_lotte_result_round_and_other_round() -> None:
    p = _pred("r1")
    p.output.lotte[1].player_name, p.output.lotte[2].player_name = "ロッテ 三位", "ロッテ 二位"  # 2位と3位を入れ替える
    table = lotte_result_table({"m1": [p]}, ACTUAL)
    assert table.rows[0] == ["1位", "ロッテ 一位・投手", "○ 1/1"]
    assert table.rows[1][2] == "△ 別の順位で1回"
    assert table.rows[4] == ["5位", "（指名なし）", "—"]  # ダミーの正解はロッテ4位まで


def test_summary_includes_baseline_row() -> None:
    scores = [score_run(_pred("r1"), ACTUAL, Master([]), {}), score_run(_pred("r2", miss_g=True), ACTUAL, Master([]), {})]
    table = summary_table(scores, BASE, ACTUAL)
    assert table.rows[0][2] == "11.5（11〜12）"
    assert table.rows[-1][:3] == ["—", BASELINE_LABEL, "0（1球団中）"]


def test_sections_per_condition() -> None:
    keys = [s.key for s in sections([_pred("r1")], ACTUAL, Master([]), {}, BASE)]
    assert keys == ["results_summary", "results_first_round_D", "results_lotte_D"]
