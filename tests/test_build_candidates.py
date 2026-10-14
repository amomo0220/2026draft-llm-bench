from pathlib import Path

import pytest

from src.build_candidates import Source, build, normalize_name, parse_submissions, write_csv
from src.candidates import COLUMNS, Category, load_candidates


def _html(sections: dict[str, list[list[str]]]) -> str:
    parts = ["<html><body><h1>一覧</h1>"]
    for heading, rows in sections.items():
        parts.append(f"<h2>{heading}</h2><table><tr><th>都道府県</th><th>学校名</th><th>氏　名</th><th>受付日</th></tr>")
        for r in rows:
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)


HS_HTML = _html(
    {
        "北海道・東北地区": [["北海道", "A高", "山田　太郎", "9月1日"], []],
        "関東・東京地区": [["神奈川", "横浜", '<a href="https://example.com/p">織田 翔希</a>', "9月17日"]],
        "NPBドラフト対象外": [["東京", "C高", "対象外 選手", "9月1日"]],
    }
)


def _source(tmp_path: Path, html: str, category: Category = Category.HIGH_SCHOOL) -> Source:
    path = tmp_path / f"{category.name}.html"
    path.write_text(html, encoding="utf-8")
    return Source(category, path, "https://example.com/list")


def test_normalize_name() -> None:
    assert normalize_name(" 山田　 太郎 ") == "山田 太郎"


def test_parse_skips_header_empty_and_excluded_section() -> None:
    assert parse_submissions(HS_HTML) == [("A高", "山田 太郎"), ("横浜", "織田 翔希")]


def test_parse_rejects_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="列数"):
        parse_submissions(_html({"近畿地区": [["京都", "B高", "選手"]]}))


def test_build_assigns_ids_and_writes_valid_csv(tmp_path: Path) -> None:
    out = tmp_path / "candidates.csv"
    rows, warnings = build([_source(tmp_path, HS_HTML)], tmp_path / "none.csv", out)
    assert [r.player_id for r in rows] == ["HS-001", "HS-002"]
    assert warnings == []
    write_csv(rows, out)
    assert load_candidates(out) == rows


def test_build_keeps_existing_ids(tmp_path: Path) -> None:
    out = tmp_path / "candidates.csv"
    rows, _ = build([_source(tmp_path, HS_HTML)], tmp_path / "none.csv", out)
    write_csv(rows, out)

    # 1人目が取り下げ、新しい選手が追加された場合
    updated = _html(
        {
            "関東・東京地区": [
                ["神奈川", "横浜", "織田 翔希", "9月17日"],
                ["千葉", "D高", "新人 選手", "10月1日"],
            ]
        }
    )
    rows2, _ = build([_source(tmp_path, updated)], tmp_path / "none.csv", out)
    assert {r.name: r.player_id for r in rows2} == {"織田 翔希": "HS-002", "新人 選手": "HS-003"}


def test_build_dedupes_identical_rows_in_source(tmp_path: Path) -> None:
    html = _html({"関東・東京地区": [["〃", "E大学", "今津　慶介", "9月24日"]] * 2})
    rows, warnings = build([_source(tmp_path, html, Category.UNIVERSITY)], tmp_path / "none.csv", tmp_path / "o.csv")
    assert [r.player_id for r in rows] == ["UN-001"]
    assert len(warnings) == 1


def test_build_merges_extra(tmp_path: Path) -> None:
    extra = tmp_path / "extra.csv"
    extra.write_text(
        ",".join(COLUMNS) + "\nSH-001,佐藤 次郎,,社会人,ダミー製作所,投手,右投右打,https://example.com/sh\n",
        encoding="utf-8",
    )
    rows, _ = build([_source(tmp_path, HS_HTML)], extra, tmp_path / "o.csv")
    assert [r.player_id for r in rows] == ["HS-001", "HS-002", "SH-001"]


def test_build_rejects_duplicate_between_source_and_extra(tmp_path: Path) -> None:
    extra = tmp_path / "extra.csv"
    extra.write_text(
        ",".join(COLUMNS) + "\nHS-001,山田 太郎,,高校,別の高校,,,https://example.com/x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="player_id が重複"):
        build([_source(tmp_path, HS_HTML)], extra, tmp_path / "o.csv")
