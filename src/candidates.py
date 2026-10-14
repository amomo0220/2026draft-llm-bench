"""候補者マスタ（data/candidates.csv）のスキーマと読み込み・検証（docs/PLAN.md 3.4）。

使い方:
    uv run python -m src.candidates data/candidates.csv
"""

import csv
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator, model_validator

COLUMNS = ["player_id", "name", "kana", "category", "team", "position", "throws_bats", "source_url"]

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Category(StrEnum):
    HIGH_SCHOOL = "高校"
    UNIVERSITY = "大学"
    SHAKAIJIN = "社会人"
    INDEPENDENT = "独立"


# player_id の接頭辞（例: HS-001）
ID_PREFIX: dict[Category, str] = {
    Category.HIGH_SCHOOL: "HS",
    Category.UNIVERSITY: "UN",
    Category.SHAKAIJIN: "SH",
    Category.INDEPENDENT: "IN",
}


def _blank_to_none(v: object) -> object:
    if isinstance(v, str) and not v.strip():
        return None
    return v


class Candidate(BaseModel):
    """candidates.csv の1行。"""

    model_config = ConfigDict(extra="forbid")

    player_id: Annotated[str, Field(pattern=r"^(HS|UN|SH|IN)-\d{3}$")]
    name: NonEmptyStr
    kana: str | None = None
    category: Category
    team: NonEmptyStr  # 所属（高校名・大学名・企業名・球団名）
    position: str | None = None
    throws_bats: str | None = None  # 例: 右投左打
    source_url: HttpUrl

    _optional = field_validator("kana", "position", "throws_bats", mode="before")(_blank_to_none)

    @model_validator(mode="after")
    def _check_id_prefix(self) -> "Candidate":
        expected = ID_PREFIX[self.category]
        if not self.player_id.startswith(expected + "-"):
            raise ValueError(f"{self.player_id}: category={self.category} の接頭辞は {expected} です")
        return self


def load_candidates(path: Path) -> list[Candidate]:
    """CSV を読み込み、全行とファイル全体の制約を検証する。問題があれば ValueError。"""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"列が違います: {reader.fieldnames}（期待値: {COLUMNS}）")
        errors: list[str] = []
        rows: list[Candidate] = []
        for lineno, raw in enumerate(reader, start=2):
            try:
                rows.append(Candidate.model_validate(raw))
            except ValueError as e:
                errors.append(f"{path}:{lineno}: {e}")

    for pid, n in Counter(r.player_id for r in rows).items():
        if n > 1:
            errors.append(f"player_id が重複しています: {pid}")
    # 同姓同名はありうるため、名前と所属の組で重複を見る
    for (name, team), n in Counter(("".join(r.name.split()), r.team) for r in rows).items():
        if n > 1:
            errors.append(f"同じ選手が重複しています: {name}（{team}）")

    if errors:
        raise ValueError("\n".join(errors))
    return rows


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "data/candidates.csv")
    try:
        rows = load_candidates(path)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    counts = Counter(r.category for r in rows)
    print(f"{path}: {len(rows)}人 " + " ".join(f"{c.value}={counts.get(c, 0)}" for c in Category))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
