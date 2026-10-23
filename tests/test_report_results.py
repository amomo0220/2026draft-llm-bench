"""答え合わせの表（ダミーの正解データで確かめる）。"""

import copy

from src.actual import ActualDraft
from src.matching import Master
from src.report_results import (
    BASELINE_LABEL,
    first_round_result_table,
    lotte_reason_result_table,
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
    assert keys == ["results_summary", "results_first_round_D", "results_lotte_D", "results_lotte_reasons_D_m1"]


def test_lotte_reason_result_lists_every_run_with_verdict() -> None:
    p1 = _pred("r1")
    p2 = _pred("r2")
    p2.output.lotte[1].player_name = "ロッテ 三位"  # 実際は3位の選手を2位に挙げた
    p2.output.lotte[2].player_name = "別の 捕手"
    p2.output.lotte[2].position = p2.output.lotte[1].position  # 捕手（実際の3位は内野手）
    p2.output.lotte[3].player_name = "別の 外野手"  # ポジションは実際の4位と同じ外野手
    table = lotte_reason_result_table([p1, p2], ACTUAL)
    assert table.header == ["順位", "実際の指名", "AI の予想", "判定", "AI が書いた理由"]
    assert table.rows[0][:4] == ["1位", "ロッテ 一位・投手", "1回目：ロッテ 一位・投手", "○ 的中"]
    assert table.rows[1][:2] == ["", ""]  # 同じ順位の2回目は、順位と実際の指名を省く
    assert table.rows[3][3] == "△ 別の順位で獲得"  # 2位に挙げた「ロッテ 三位」
    assert table.rows[5][3] == "×"  # 3位の「別の 捕手」
    assert table.rows[7][3] == "ポジション一致"  # 4位の「別の 外野手」
    assert table.rows[8][1] == "（指名なし）"  # ダミーの正解はロッテ4位まで
    assert table.rows[0][4] == "ダミー" and table.highlight[0] is True and table.highlight[5] is False
    assert len(table.rows) == 10  # 5順位 × 2回
