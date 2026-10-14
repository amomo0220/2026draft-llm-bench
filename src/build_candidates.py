"""プロ志望届の一覧ページ（data/raw/ に保存した HTML）から data/candidates.csv を生成する。

生データ（HTML）はコミットしない。コミットするのは生成した CSV とこのスクリプトのみ。
社会人・独立リーグの候補は data/candidates_extra.csv に手で入力し、ここで結合する。

player_id は再生成しても変わらないようにする:
既存の candidates.csv にある選手（区分・所属・名前が一致）は同じ ID を使い、
新しい選手には区分ごとの最大番号の次を振る。

使い方:
    uv run python -m src.build_candidates
"""

import csv
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from src.candidates import COLUMNS, ID_PREFIX, Candidate, Category, check_uniqueness, load_candidates, name_key


@dataclass(frozen=True)
class Source:
    category: Category
    html_path: Path
    url: str


SOURCES = [
    Source(Category.HIGH_SCHOOL, Path("data/raw/bc_287000_highschool.html"), "https://www.baseballchannel.jp/npb/287000/"),
    Source(Category.UNIVERSITY, Path("data/raw/bc_287011_university.html"), "https://www.baseballchannel.jp/npb/287011/"),
]
EXTRA_PATH = Path("data/candidates_extra.csv")
OUT_PATH = Path("data/candidates.csv")

# この見出しの下の表は NPB ドラフトの対象外なので読まない
EXCLUDED_SECTION = "NPBドラフト対象外"
HEADER_FIRST_CELLS = {"都道府県", "連盟"}
NUM_CELLS = 4  # 都道府県（連盟）／学校名／氏名／受付日


class _TableParser(HTMLParser):
    """h2 見出しごとに、その下の表の行（セル文字列のリスト）を集める。"""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[tuple[str, list[list[str]]]] = []
        self._in_h2 = False
        self._h2_buf: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h2":
            self._in_h2, self._h2_buf = True, []
        elif tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._in_h2:
            self._in_h2 = False
            self.sections.append(("".join(self._h2_buf).strip(), []))
        elif tag == "table":
            self._in_table = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self.sections:
                self.sections[-1][1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_h2:
            self._h2_buf.append(data)
        elif self._cell is not None:
            self._cell.append(data)


def normalize_name(name: str) -> str:
    """空白（全角を含む）を半角スペース1つにそろえる。"""
    return " ".join(name.split())


def parse_submissions(html: str) -> list[tuple[str, str]]:
    """志望届一覧の HTML から (学校名, 氏名) のリストを返す。"""
    parser = _TableParser()
    parser.feed(html)
    result: list[tuple[str, str]] = []
    for heading, rows in parser.sections:
        if heading == EXCLUDED_SECTION:
            continue
        for cells in rows:
            if not cells or cells[0] in HEADER_FIRST_CELLS:
                continue
            if len(cells) != NUM_CELLS:
                raise ValueError(f"{heading}: 列数が想定外です: {cells}")
            result.append((cells[1].strip(), normalize_name(cells[2])))
    return result


def _id_number(player_id: str) -> int:
    return int(player_id.split("-")[1])


def build(sources: list[Source], extra_path: Path, out_path: Path) -> tuple[list[Candidate], list[str]]:
    """候補者を組み立て、(行, 警告) を返す。出典内の完全一致の重複行は1件にまとめて警告する。"""
    existing = load_candidates(out_path) if out_path.exists() else []
    known = {(c.category, c.team, name_key(c.name)): c.player_id for c in existing}
    next_no = {cat: 1 + max((_id_number(c.player_id) for c in existing if c.category == cat), default=0) for cat in Category}

    rows: list[Candidate] = []
    warnings: list[str] = []
    for src in sources:
        seen: set[tuple[str, str]] = set()
        for team, name in parse_submissions(src.html_path.read_text(encoding="utf-8")):
            if (team, name_key(name)) in seen:
                warnings.append(f"出典内の重複行を1件にまとめました: {name}（{team}） {src.url}")
                continue
            seen.add((team, name_key(name)))
            pid = known.get((src.category, team, name_key(name)))
            if pid is None:
                pid = f"{ID_PREFIX[src.category]}-{next_no[src.category]:03d}"
                next_no[src.category] += 1
            rows.append(Candidate(player_id=pid, name=name, category=src.category, team=team, source_url=src.url))

    if extra_path.exists():
        rows += load_candidates(extra_path)

    errors = check_uniqueness(rows)
    if errors:
        raise ValueError("\n".join(errors))
    return rows, warnings


def write_csv(rows: list[Candidate], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            d = r.model_dump(mode="json")
            w.writerow({k: "" if d[k] is None else d[k] for k in COLUMNS})


def main() -> int:
    try:
        rows, warnings = build(SOURCES, EXTRA_PATH, OUT_PATH)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    for w in warnings:
        print(f"警告: {w}", file=sys.stderr)
    write_csv(rows, OUT_PATH)
    load_candidates(OUT_PATH)  # 書き出した結果も検証する
    counts = {c: sum(r.category == c for r in rows) for c in Category}
    print(f"{OUT_PATH}: {len(rows)}人 " + " ".join(f"{c.value}={n}" for c, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
