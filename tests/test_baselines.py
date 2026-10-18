"""メディア予想のベースライン（架空の媒体・選手で確かめる）。"""

import csv
from pathlib import Path

import pytest

from src.baselines import COLUMNS, check, consensus, load_media_picks, main
from src.scorer import brier, top_candidate


def _write(path: Path, rows: list[dict[str, str]], columns: list[str] = COLUMNS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def _row(outlet: str, team: str, name: str, published_on: str = "2026-10-15") -> dict[str, str]:
    return {
        "outlet": outlet,
        "published_on": published_on,
        "source_url": "https://example.com/a",
        "team": team,
        "player_name": name,
        "affiliation": "ダミー大",
        "position": "投手",
    }


ROWS = [_row("A紙", "G", "甲 太郎"), _row("B紙", "G", "甲太郎"), _row("C紙", "G", "乙 次郎"), _row("A紙", "M", "丙 三郎")]


def test_consensus_shares_by_outlet(tmp_path: Path) -> None:
    picks = load_media_picks(_write(tmp_path / "m.csv", ROWS))
    bids = {b.team.value: b for b in consensus(picks)}
    assert set(bids) == {"G", "M"}  # 予想のない球団は含めない
    g = bids["G"]
    # 空白の違いは同じ選手として数える。確率は、その球団を予想した3媒体のうちの割合
    assert [(c.player_name, round(c.prob, 3)) for c in g.candidates] == [("甲 太郎", 0.667), ("乙 次郎", 0.333)]
    assert top_candidate(g).player_name == "甲 太郎"
    assert brier(g, "甲太郎") == pytest.approx((1 / 3) ** 2 + (1 / 3) ** 2)


def test_duplicate_team_in_outlet_is_error(tmp_path: Path) -> None:
    picks = load_media_picks(_write(tmp_path / "m.csv", [*ROWS, _row("A紙", "G", "丁 四郎")]))
    assert check(picks) == ["A紙 の G が2行あります"]


def test_invalid_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="列が違います"):
        load_media_picks(_write(tmp_path / "a.csv", [], columns=COLUMNS[:-1]))
    with pytest.raises(ValueError, match="m.csv:2"):
        load_media_picks(_write(tmp_path / "m.csv", [{**ROWS[0], "position": "遊撃手"}]))


def test_main_marks_late_articles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path / "m.csv", [*ROWS, _row("D紙", "T", "戊 五郎", published_on="2026-10-21")])
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "4媒体、5行" in out
    assert "D紙: 1球団、公開日 2026-10-21 ※AI の予想日より後" in out
    assert "G: 甲 太郎（67%）" in out
