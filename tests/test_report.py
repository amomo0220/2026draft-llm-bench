"""記事用のまとめ（ダミーの予測で確かめる）。"""

import copy
import json
from pathlib import Path

from src.candidates import Candidate
from src.matching import Master
from src.report import (
    build_report,
    contest_table,
    first_round_table,
    lotte_reason_table,
    lotte_table,
    main,
)
from src.run import RunLog
from src.schema import Prediction
from tests.dummy_draft import prediction_payload


def _pred(run_id: str, model_id: str = "m1", condition: str = "D", lotte1: str | None = None) -> Prediction:
    payload = copy.deepcopy(prediction_payload(run_id, model_id, condition))
    if lotte1:
        payload["output"]["lotte"][0]["player_name"] = lotte1
    return Prediction.model_validate(payload)


def _log(cond: str, model: str, cost: float, status: str = "ok") -> RunLog:
    return RunLog(
        run_id="r", model_id=model, provider="p", condition=cond, run_no=1, prompt_sha256="x", started_at="t",
        duration_sec=10, attempts=1, input_tokens=1, output_tokens=1, reasoning_tokens=None, cost_usd=cost,
        tool_calls=2, function_calls=4, api_retries=0, status=status,
    )  # fmt: skip


def test_first_round_counts_top_pick_per_run() -> None:
    table = str(first_round_table({"m1": [_pred("r1"), _pred("r2")]}))
    assert "| 巨人 | 競合 一郎・投手（2/2） |" in table
    assert table.splitlines()[-1] == "| ロッテ | ロッテ 一位・投手（2/2） |"  # セ → パの順で、ロッテが最後


def test_contest_lists_players_with_two_or_more_teams() -> None:
    table = str(contest_table({"m1": [_pred("r1")], "m2": [_pred("r2", "m2")]}))
    assert "| 競合 一郎 | 2.0 | 2.0 |" in table  # G と T が最有力に挙げる
    assert "選手H" not in table  # 1球団だけの選手は出さない


def test_lotte_shows_mode_and_agreement() -> None:
    table = str(lotte_table({"m1": [_pred("r1"), _pred("r2", lotte1="別の 選手"), _pred("r3")]}))
    assert "| 1位 | ロッテ 一位・投手（2/3） |" in table


def test_report_counts_names_missing_from_master() -> None:
    master = Master([Candidate(player_id="HS-001", name="競合 一郎", category="高校", team="ダミー高", source_url="https://example.com")])
    report = build_report([_pred("r1")], [_log("D", "m1", 1.5), _log("D", "m1", 0.5, "error")], master)
    assert "## 12球団の1位入札予想（条件D「エージェント」）" in report
    # 登場する選手: 1位の最有力11人（「競合 一郎」は2球団）＋次点12人＋ロッテ2〜5位の4人（1位は最有力と同じ）= 27人。
    # マスタにあるのは「競合 一郎」だけ
    assert "| D | m1 | 1 | 27 | 26 |" in report
    assert "| D | m1 | 1/2 | $1.00 |" in report and "合計 2 回、$2.00" in report


def test_main_without_predictions(tmp_path: Path) -> None:
    assert main(["--predictions", str(tmp_path)]) == 1


def test_main_prints_report(tmp_path: Path, capsys) -> None:
    out = tmp_path / "p" / "D" / "m1"
    out.mkdir(parents=True)
    (out / "r1.json").write_text(_pred("r1").model_dump_json(), encoding="utf-8")
    (tmp_path / "p" / "late").mkdir()
    (tmp_path / "p" / "late" / "x.json").write_text(json.dumps({}), encoding="utf-8")  # late/ は読まない
    assert main(["--predictions", str(tmp_path / "p"), "--runs", str(tmp_path / "none.jsonl")]) == 0
    assert "条件D「エージェント」" in capsys.readouterr().out


def test_first_round_highlights_rows_where_all_models_agree() -> None:
    other = Prediction.model_validate(
        {**prediction_payload("r2", "m2"), "output": {**prediction_payload()["output"], "first_round": [
            {**b, "candidates": [{**b["candidates"][0], "prob": 0.4}, {**b["candidates"][1], "prob": 0.6}]} if b["team"] == "G" else b
            for b in prediction_payload()["output"]["first_round"]
        ]}}
    )  # fmt: skip
    table = first_round_table({"m1": [_pred("r1")], "m2": [other]})
    assert table.header == ["球団", "m1", "m2"]
    assert table.highlight[0] is False  # 巨人だけ m2 の本命が違う
    assert all(table.highlight[1:])


def test_model_labels_in_header() -> None:
    table = lotte_table({"claude-sonnet-5-5": [_pred("r1", "claude-sonnet-5-5")]})
    assert table.header == ["順位", "Claude Sonnet 5.5"]


def test_empty_contest_table_prints_message() -> None:
    p = Prediction.model_validate(
        {**prediction_payload(), "output": {**prediction_payload()["output"], "first_round": [
            {**b, "candidates": [{**b["candidates"][0], "player_name": f"単独{b['team']}", "prob": 1.0}]}
            for b in prediction_payload()["output"]["first_round"]
        ]}}
    )  # fmt: skip
    assert str(contest_table({"m1": [p]})) == "（入札が重なると予想された選手はいません）"


def test_lotte_reason_table_uses_reason_of_mode_pick() -> None:
    p1 = _pred("r1", lotte1="別の 選手")
    p1.output.lotte[0].reason = "別の理由"
    p2, p3 = _pred("r2"), _pred("r3")
    p2.output.lotte[0].reason = "2回目の理由"
    table = lotte_reason_table([p1, p2, p3])
    assert table.header == ["順位", "予想した選手", "AI が書いた理由"]
    # 1位は「ロッテ 一位」が3回中2回。理由は、その選手を挙げた最初の実行（r2）のもの
    assert table.rows[0] == ["1位", "ロッテ 一位・投手（2/3）", "2回目の理由"]
    assert table.rows[1] == ["2位", "ロッテ 二位・捕手（3/3）", "ダミー"]


def test_report_has_reason_section_per_model() -> None:
    master = Master([])
    report = build_report([_pred("r1"), _pred("r2", "m2")], [], master)
    assert "## ロッテの1〜5位予想と理由：m1（条件D「エージェント」）" in report
    assert "## ロッテの1〜5位予想と理由：m2（条件D「エージェント」）" in report
