"""記事用のまとめ（ダミーの予測で確かめる）。"""

import copy
import json
from pathlib import Path

from src.candidates import Candidate
from src.matching import Master
from src.report import build_report, contest_table, first_round_table, lotte_table, main
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
    table = first_round_table({"m1": [_pred("r1"), _pred("r2")]})
    assert "| 巨人 | 競合 一郎（2/2） |" in table
    assert table.splitlines()[-1] == "| ロッテ | ロッテ 一位（2/2） |"  # セ → パの順で、ロッテが最後


def test_contest_lists_players_with_two_or_more_teams() -> None:
    table = contest_table({"m1": [_pred("r1")], "m2": [_pred("r2", "m2")]})
    assert "| 競合 一郎 | 2.0 | 2.0 |" in table  # G と T が最有力に挙げる
    assert "選手H" not in table  # 1球団だけの選手は出さない


def test_lotte_shows_mode_and_agreement() -> None:
    table = lotte_table({"m1": [_pred("r1"), _pred("r2", lotte1="別の 選手"), _pred("r3")]})
    assert "| 1位 | ロッテ 一位（2/3） |" in table


def test_report_counts_names_missing_from_master() -> None:
    master = Master([Candidate(player_id="HS-001", name="競合 一郎", category="高校", team="ダミー高", source_url="https://example.com")])
    report = build_report([_pred("r1")], [_log("D", "m1", 1.5), _log("D", "m1", 0.5, "error")], master)
    assert "## 条件D（エージェント）" in report
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
    assert "## 条件D" in capsys.readouterr().out
