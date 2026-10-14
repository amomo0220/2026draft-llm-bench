"""プロ志望届の公式一覧（data/raw/ に保存した HTML）から data/candidates.csv を生成する。

出典:
- 高校: 日本高等学校野球連盟（jhbf.or.jp）
- 大学: 全日本大学野球連盟（jubf.net）

生データ（HTML）はコミットしない。コミットするのは生成した CSV とこのスクリプトのみ。
社会人・独立リーグの候補は data/candidates_extra.csv に手で入力し、ここで結合する。

読み込むのは見出しが「NPBドラフト対象者」の表だけ。ページに書かれた「合計：N名」と
読み込んだ人数が一致しない場合はエラーにする。

player_id は再生成しても変わらないようにする:
既存の candidates.csv にある選手は同じ ID を使う（区分・所属・名前が一致する選手、
所属が変わった場合は区分・名前が一意に一致する選手）。新しい選手には区分ごとの最大番号の次を振る。

使い方:
    uv run python -m src.build_candidates
"""

import csv
import re
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
    num_cols: int
    team_col: int
    name_col: int
    kana_col: int | None = None


SOURCES = [
    # 都道府県／学校名／氏名／受付日
    Source(
        Category.HIGH_SCHOOL,
        Path("data/raw/jhbf_2026_highschool.html"),
        "https://www.jhbf.or.jp/pro-aspiring/2026.html",
        num_cols=4,
        team_col=1,
        name_col=2,
    ),
    # 連盟／学校名／氏名／ふりがな／受付日
    Source(
        Category.UNIVERSITY,
        Path("data/raw/jubf_2026_university.html"),
        "https://www.jubf.net/system/prog/procandidate.php?kind=all&year=2026",
        num_cols=5,
        team_col=1,
        name_col=2,
        kana_col=3,
    ),
]
EXTRA_PATH = Path("data/candidates_extra.csv")
OUT_PATH = Path("data/candidates.csv")

TARGET_SECTION = "NPBドラフト対象者"
HEADER_FIRST_CELLS = {"都道府県", "連盟"}
TOTAL_RE = re.compile(r"合計[：:\s]*(\d+)\s*名")


@dataclass
class _Section:
    heading: str
    rows: list[list[str]]
    text: list[str]


class _SectionParser(HTMLParser):
    """見出し（h1〜h4、または class="section" の span）ごとに、表の行と本文テキストを集める。"""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[_Section] = []
        self._heading_tag: str | None = None
        self._heading_buf: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        is_section_span = tag == "span" and dict(attrs).get("class") == "section"
        if tag in ("h1", "h2", "h3", "h4") or is_section_span:
            self._heading_tag, self._heading_buf = tag, []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == self._heading_tag:
            self._heading_tag = None
            self.sections.append(_Section("".join(self._heading_buf).strip(), [], []))
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self.sections and self._row:
                self.sections[-1].rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._heading_tag:
            self._heading_buf.append(data)
            return
        if self._cell is not None:
            self._cell.append(data)
        if self.sections:
            self.sections[-1].text.append(data)


def normalize_space(text: str) -> str:
    """空白（全角を含む）を半角スペース1つにそろえる。"""
    return " ".join(text.split())


@dataclass(frozen=True)
class Submission:
    team: str
    name: str
    kana: str | None


def parse_submissions(html: str, src: Source) -> list[Submission]:
    """志望届一覧の HTML から「NPBドラフト対象者」の行を返す。"""
    parser = _SectionParser()
    parser.feed(html)
    targets = [s for s in parser.sections if s.heading.endswith(TARGET_SECTION)]
    if len(targets) != 1:
        raise ValueError(f"{src.url}: 「{TARGET_SECTION}」の見出しが {len(targets)} 個あります")
    section = targets[0]

    result: list[Submission] = []
    for cells in section.rows:
        if cells[0] in HEADER_FIRST_CELLS or "合計" in cells:
            continue  # 見出し行・合計行
        if len(cells) == src.num_cols and not cells[src.team_col] and not cells[src.name_col]:
            continue  # 提出者のいない連盟の行（連盟名だけが入っている）
        if len(cells) != src.num_cols:
            raise ValueError(f"{src.url}: 列数が想定外です: {cells}")
        kana = normalize_space(cells[src.kana_col]) if src.kana_col is not None else None
        result.append(Submission(cells[src.team_col].strip(), normalize_space(cells[src.name_col]), kana or None))

    m = TOTAL_RE.search("".join(section.text))
    if m is None:
        raise ValueError(f"{src.url}: 「合計：N名」が見つかりません")
    if int(m.group(1)) != len(result):
        raise ValueError(f"{src.url}: ページの合計 {m.group(1)}名 と読み込んだ人数 {len(result)}名 が一致しません")
    return result


def _id_number(player_id: str) -> int:
    return int(player_id.split("-")[1])


class _IdAssigner:
    """既存の candidates.csv をもとに player_id を振る。"""

    def __init__(self, existing: list[Candidate]) -> None:
        self._by_team = {(c.category, c.team, name_key(c.name)): c.player_id for c in existing}
        by_name: dict[tuple[Category, str], list[str]] = {}
        for c in existing:
            by_name.setdefault((c.category, name_key(c.name)), []).append(c.player_id)
        self._by_name = {k: v[0] for k, v in by_name.items() if len(v) == 1}
        self._next = {
            cat: 1 + max((_id_number(c.player_id) for c in existing if c.category == cat), default=0) for cat in Category
        }
        self._used: set[str] = set()

    def assign(self, category: Category, team: str, name: str) -> str:
        key = name_key(name)
        pid = self._by_team.get((category, team, key)) or self._by_name.get((category, key))
        if pid is None or pid in self._used:
            pid = f"{ID_PREFIX[category]}-{self._next[category]:03d}"
            self._next[category] += 1
        self._used.add(pid)
        return pid


def build(sources: list[Source], extra_path: Path, out_path: Path) -> tuple[list[Candidate], list[str]]:
    """候補者を組み立て、(行, 警告) を返す。出典内の完全一致の重複行は1件にまとめて警告する。"""
    ids = _IdAssigner(load_candidates(out_path) if out_path.exists() else [])
    rows: list[Candidate] = []
    warnings: list[str] = []
    for src in sources:
        seen: set[tuple[str, str]] = set()
        for sub in parse_submissions(src.html_path.read_text(encoding="utf-8"), src):
            if (sub.team, name_key(sub.name)) in seen:
                warnings.append(f"出典内の重複行を1件にまとめました: {sub.name}（{sub.team}） {src.url}")
                continue
            seen.add((sub.team, name_key(sub.name)))
            rows.append(
                Candidate(
                    player_id=ids.assign(src.category, sub.team, sub.name),
                    name=sub.name,
                    kana=sub.kana,
                    category=src.category,
                    team=sub.team,
                    source_url=src.url,
                )
            )

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
