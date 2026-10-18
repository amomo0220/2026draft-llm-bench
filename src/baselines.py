"""メディア予想のベースライン（docs/PLAN.md 3.3）。

`data/baselines/media_picks.csv` に、媒体ごとの「12球団の1位予想」を1行＝1媒体の1球団で入力する。
ここでは検証と、球団ごとの多数決（媒体の数で割った割合を確率とみなす）だけを行う。
多数決の結果は予測と同じ `FirstRoundBid` の形にするので、採点は src/scorer.py の関数（top_candidate・brier）をそのまま使える。

    uv run python -m src.baselines data/baselines/media_picks.csv
"""

import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, HttpUrl, ValidationError

from src.matching import normalize_name
from src.schema import FirstRoundBid, NonEmptyStr, PlayerProb, Position, TeamCode

COLUMNS = ["outlet", "published_on", "source_url", "team", "player_name", "affiliation", "position"]
# AI の予想（1週間前予想）を実行した日。これより大きく後の予想は、条件をそろえた比較にならない
AI_PREDICTION_DATE = date(2026, 10, 15)


class MediaPick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outlet: NonEmptyStr  # 媒体名（例: スポーツナビ）。同じ媒体の別記事は「媒体名（記事の版）」で区別する
    published_on: date  # 記事の公開日（更新日が分かればその日）
    source_url: HttpUrl
    team: TeamCode
    player_name: NonEmptyStr
    affiliation: NonEmptyStr
    position: Position


def load_media_picks(path: Path) -> list[MediaPick]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"列が違います: {reader.fieldnames}（期待: {COLUMNS}）")
        rows = []
        for i, row in enumerate(reader, start=2):
            try:
                rows.append(MediaPick.model_validate(row))
            except ValidationError as e:
                raise ValueError(f"{path}:{i}: {e}") from e
    return rows


def check(picks: list[MediaPick]) -> list[str]:
    """入力の誤り（同じ媒体で同じ球団が2行）。"""
    seen: set[tuple[str, TeamCode]] = set()
    errors = []
    for p in picks:
        key = (p.outlet, p.team)
        if key in seen:
            errors.append(f"{p.outlet} の {p.team.value} が2行あります")
        seen.add(key)
    return errors


def consensus(picks: list[MediaPick]) -> list[FirstRoundBid]:
    """球団ごとの多数決。確率は、その球団を予想した媒体のうち、その選手を挙げた割合。予想のない球団は含めない。"""
    by_team: dict[TeamCode, list[MediaPick]] = defaultdict(list)
    for p in picks:
        by_team[p.team].append(p)
    bids = []
    for team in TeamCode:
        rows = by_team.get(team)
        if not rows:
            continue
        counts: dict[str, list[MediaPick]] = {}
        for p in rows:
            counts.setdefault(normalize_name(p.player_name), []).append(p)
        # 多い順。同数なら先に出た順（CSV の順）
        ordered = sorted(counts.values(), key=lambda ps: -len(ps))
        candidates = [
            PlayerProb(
                player_name=ps[0].player_name, affiliation=ps[0].affiliation, position=ps[0].position, prob=len(ps) / len(rows)
            )
            for ps in ordered
        ]
        bids.append(FirstRoundBid(team=team, candidates=candidates))
    return bids


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[0])
    picks = load_media_picks(path)
    errors = check(picks)
    for e in errors:
        print(f"NG: {e}", file=sys.stderr)
    outlets = sorted({p.outlet for p in picks})
    print(f"{path}: {len(outlets)}媒体、{len(picks)}行")
    for o in outlets:
        rows = [p for p in picks if p.outlet == o]
        dates = sorted({p.published_on for p in rows})
        late = " ※AI の予想日より後" if dates[-1] > AI_PREDICTION_DATE else ""
        print(f"  {o}: {len(rows)}球団、公開日 {', '.join(map(str, dates))}{late}")
    for bid in consensus(picks):
        top = max(bid.candidates, key=lambda c: c.prob)
        print(f"  {bid.team.value}: {top.player_name}（{top.prob:.0%}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
