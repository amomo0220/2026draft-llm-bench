from pathlib import Path

import pytest

from src.build_candidates import Source, Submission, build, normalize_space, parse_submissions, write_csv
from src.candidates import COLUMNS, Category, load_candidates


def _hs_html(rows: list[list[str]], total: int | None = None) -> str:
    """高野連の一覧ページを模した HTML（h4 見出し、合計は表の外）。"""
    total = len(rows) if total is None else total
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        "<html><body><h3>2026年 プロ野球志望届提出者一覧</h3>"
        "<h4>NPBドラフト対象外</h4><table><tr><th>都道府県</th><th>学校名</th><th>氏　名</th><th>受付日</th></tr>"
        "<tr><td>山梨</td><td>X高</td><td>対象外 選手</td><td>10月10日</td></tr></table><p>合計：1名</p>"
        "<h4>NPBドラフト対象者</h4><table><tr><th>都道府県</th><th>学校名</th><th>氏　名</th><th>受付日</th></tr>"
        f"{body}</table><p>合計：{total}名</p></body></html>"
    )


def _un_html(rows: list[list[str]], total: int) -> str:
    """大学野球連盟の一覧ページを模した HTML（span.section 見出し、合計は表の中）。"""
    body = "".join("<tr>" + "".join(f"<td><font>{c}</font></td>" for c in r) + "</tr>" for r in rows)
    return (
        '<html><body><span class="chapter">2026年度</span>'
        '<span class="section"><font>■</font>NPBドラフト対象者</span>'
        "<table><tr><td>連盟</td><td>学校名</td><td>氏名</td><td>ふりがな</td><td>所属連盟<br>受付日</td></tr>"
        f"{body}<tr><td></td><td></td><td>合計</td><td>{total}名</td><td></td></tr></table>"
        '<span class="section"><font>■</font>NPBドラフト対象外者</span>'
        "<table><tr><td>連盟</td><td>学校名</td><td>氏名</td><td>ふりがな</td><td>受付日</td></tr>"
        "<tr><td>A連盟</td><td>Z大学</td><td>対象外 選手</td><td>たいしょうがい</td><td>9月1日</td></tr></table>"
        "</body></html>"
    )


def _hs_source(path: Path) -> Source:
    return Source(Category.HIGH_SCHOOL, path, "https://example.com/hs", num_cols=4, team_col=1, name_col=2)


def _un_source(path: Path) -> Source:
    return Source(Category.UNIVERSITY, path, "https://example.com/un", num_cols=5, team_col=1, name_col=2, kana_col=3)


def _write(tmp_path: Path, name: str, html: str) -> Path:
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return path


HS_ROWS = [
    ["北海道", "A高", "山田　太郎", "9月1日"],
    ["神奈川", "横浜", '<a href="https://example.com/p">織田 翔希</a>', "9月17日"],
]


def test_normalize_space() -> None:
    assert normalize_space(" 山田　 太郎 ") == "山田 太郎"


def test_parse_high_school_reads_only_target_section() -> None:
    subs = parse_submissions(_hs_html(HS_ROWS), _hs_source(Path("x")))
    assert subs == [Submission("A高", "山田 太郎", None), Submission("横浜", "織田 翔希", None)]


def test_parse_university_skips_empty_league_and_total_rows() -> None:
    rows = [
        ["A連盟", "B大学", "鈴木　一郎", "すずき　いちろう", "9月1日"],
        ["C連盟", "", "", "", ""],  # 提出者のいない連盟
        ["〃", "D大学", "佐藤 次郎", "さとう じろう", "9月2日"],
    ]
    subs = parse_submissions(_un_html(rows, total=2), _un_source(Path("x")))
    assert subs == [
        Submission("B大学", "鈴木 一郎", "すずき いちろう"),
        Submission("D大学", "佐藤 次郎", "さとう じろう"),
    ]


def test_parse_rejects_total_mismatch() -> None:
    with pytest.raises(ValueError, match="一致しません"):
        parse_submissions(_hs_html(HS_ROWS, total=3), _hs_source(Path("x")))


def test_parse_rejects_missing_target_section() -> None:
    with pytest.raises(ValueError, match="見出し"):
        parse_submissions("<html><h4>別の見出し</h4><table></table></html>", _hs_source(Path("x")))


def test_parse_rejects_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="列数"):
        parse_submissions(_hs_html([["京都", "B高", "選手"]]), _hs_source(Path("x")))


def test_build_assigns_ids_and_writes_valid_csv(tmp_path: Path) -> None:
    out = tmp_path / "candidates.csv"
    src = _hs_source(_write(tmp_path, "hs.html", _hs_html(HS_ROWS)))
    rows, warnings = build([src], tmp_path / "none.csv", out)
    assert [r.player_id for r in rows] == ["HS-001", "HS-002"]
    assert warnings == []
    write_csv(rows, out)
    assert load_candidates(out) == rows


def test_build_keeps_existing_ids(tmp_path: Path) -> None:
    out = tmp_path / "candidates.csv"
    rows, _ = build([_hs_source(_write(tmp_path, "hs.html", _hs_html(HS_ROWS)))], tmp_path / "none.csv", out)
    write_csv(rows, out)

    # 1人目が取り下げ、2人目の所属表記が変わり、新しい選手が追加された場合
    updated = [["神奈川", "横浜高", "織田 翔希", "9月17日"], ["千葉", "D高", "新人 選手", "10月1日"]]
    src = _hs_source(_write(tmp_path, "hs.html", _hs_html(updated)))
    rows2, _ = build([src], tmp_path / "none.csv", out)
    assert {r.name: r.player_id for r in rows2} == {"織田 翔希": "HS-002", "新人 選手": "HS-003"}


def test_build_dedupes_identical_rows_in_source(tmp_path: Path) -> None:
    row = ["〃", "E大学", "今津　慶介", "いまづ けいすけ", "9月24日"]
    src = _un_source(_write(tmp_path, "un.html", _un_html([row, row], total=2)))
    rows, warnings = build([src], tmp_path / "none.csv", tmp_path / "o.csv")
    assert [r.player_id for r in rows] == ["UN-001"]
    assert len(warnings) == 1


def test_build_merges_extra(tmp_path: Path) -> None:
    extra = _write(
        tmp_path,
        "extra.csv",
        ",".join(COLUMNS) + "\nSH-001,佐藤 次郎,,社会人,ダミー製作所,投手,右投右打,https://example.com/sh\n",
    )
    src = _hs_source(_write(tmp_path, "hs.html", _hs_html(HS_ROWS)))
    rows, _ = build([src], extra, tmp_path / "o.csv")
    assert [r.player_id for r in rows] == ["HS-001", "HS-002", "SH-001"]


def test_build_rejects_duplicate_between_source_and_extra(tmp_path: Path) -> None:
    extra = _write(
        tmp_path,
        "extra.csv",
        ",".join(COLUMNS) + "\nHS-001,山田 太郎,,高校,別の高校,,,https://example.com/x\n",
    )
    src = _hs_source(_write(tmp_path, "hs.html", _hs_html(HS_ROWS)))
    with pytest.raises(ValueError, match="player_id が重複"):
        build([src], extra, tmp_path / "o.csv")
