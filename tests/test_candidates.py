import csv
from pathlib import Path

import pytest

from src.candidates import COLUMNS, Category, load_candidates

ROWS = [
    ["HS-001", "山田 太郎", "やまだ たろう", "高校", "ダミー高校", "投手", "右投右打", "https://example.com/hs"],
    ["UN-001", "鈴木 一郎", "", "大学", "ダミー大学", "", "", "https://example.com/un"],
    ["SH-001", "佐藤 次郎", "", "社会人", "ダミー製作所", "内野手", "", "https://example.com/sh"],
    ["IN-001", "田中 三郎", "", "独立", "ダミー独立球団", "外野手", "左投左打", "https://example.com/in"],
    ["FM-001", "高橋 四郎", "", "二軍参加", "ダミー球団", "投手", "", "https://example.com/fm"],
]


def _write(tmp_path: Path, rows: list[list[str]], columns: list[str] = COLUMNS) -> Path:
    path = tmp_path / "candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    return path


def test_valid_file(tmp_path: Path) -> None:
    rows = load_candidates(_write(tmp_path, ROWS))
    assert [r.category for r in rows] == list(Category)
    assert rows[1].kana is None and rows[1].position is None  # 空欄は None


def test_wrong_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="列が違います"):
        load_candidates(_write(tmp_path, ROWS, columns=COLUMNS[:-1] + ["url"]))


@pytest.mark.parametrize(
    ("col", "value"),
    [
        (0, "HS-1"),  # 連番は3桁
        (0, "XX-001"),  # 未知の接頭辞
        (0, "UN-002"),  # category=高校 なのに UN
        (1, " "),  # 名前が空
        (3, "中学"),  # 未知の区分
        (4, ""),  # 所属が空
        (7, "not a url"),  # 出典URLが不正
        (7, ""),  # 出典URLは必須
    ],
)
def test_invalid_row(tmp_path: Path, col: int, value: str) -> None:
    rows = [r.copy() for r in ROWS]
    rows[0][col] = value
    with pytest.raises(ValueError):
        load_candidates(_write(tmp_path, rows))


def test_duplicate_player_id(tmp_path: Path) -> None:
    rows = ROWS + [["HS-001", "別人", "", "高校", "別の高校", "", "", "https://example.com/x"]]
    with pytest.raises(ValueError, match="player_id が重複"):
        load_candidates(_write(tmp_path, rows))


def test_duplicate_player_same_team(tmp_path: Path) -> None:
    rows = ROWS + [["HS-002", "山田太郎", "", "高校", "ダミー高校", "", "", "https://example.com/x"]]
    with pytest.raises(ValueError, match="同じ選手"):
        load_candidates(_write(tmp_path, rows))


def test_same_name_different_team_is_ok(tmp_path: Path) -> None:
    rows = ROWS + [["HS-002", "山田 太郎", "", "高校", "別の高校", "", "", "https://example.com/x"]]
    assert len(load_candidates(_write(tmp_path, rows))) == len(ROWS) + 1
