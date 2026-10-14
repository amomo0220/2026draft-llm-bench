"""ドラフト当日の実際の結果（data/actual/*.json）のスキーマと検証。

当日に人手で入力するため、入力ミスをここで検出する。ポジションは NPB 公式の指名結果の4区分で入力する。

使い方:
    uv run python -m src.actual data/actual/2026.json
"""

import sys
from pathlib import Path

from pydantic import Field, model_validator

from src.matching import normalize_name
from src.schema import LOTTE_ROUNDS, NUM_TEAMS, NonEmptyStr, Position, TeamCode, _Strict


class ActualPlayer(_Strict):
    player_name: NonEmptyStr
    affiliation: NonEmptyStr
    position: Position


class ActualFirstRound(_Strict):
    team: TeamCode
    bid: ActualPlayer  # 1回目の1位入札（採点対象）
    won_lottery: bool  # 単独入札、または抽選で交渉権を得た場合 True
    final: ActualPlayer  # 最終的な1位（外れ1位を含む）。won_lottery なら bid と同じ選手

    @model_validator(mode="after")
    def _check_final(self) -> "ActualFirstRound":
        same = normalize_name(self.bid.player_name) == normalize_name(self.final.player_name)
        if self.won_lottery and not same:
            raise ValueError(f"{self.team}: won_lottery=true なら final は bid と同じ選手です")
        if not self.won_lottery and same:
            raise ValueError(f"{self.team}: 抽選で外れたのに final が bid と同じ選手です")
        return self


class ActualPick(ActualPlayer):
    round: int = Field(ge=2, le=LOTTE_ROUNDS)


class ActualDraft(_Strict):
    first_round: list[ActualFirstRound]  # 12球団
    # ロッテの2〜5位（支配下のみ）。1位は first_round の M を使う。指名しなかった順位は含めない
    lotte: list[ActualPick] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ActualDraft":
        teams = [r.team for r in self.first_round]
        if len(teams) != NUM_TEAMS or set(teams) != set(TeamCode):
            raise ValueError("first_round は12球団を1件ずつ含む必要があります")
        rounds = [p.round for p in self.lotte]
        if len(rounds) != len(set(rounds)):
            raise ValueError(f"lotte の順位が重複しています: {sorted(rounds)}")
        if sorted(rounds) != list(range(2, 2 + len(rounds))):
            raise ValueError(f"lotte は2位から欠けなく並ぶ必要があります: {sorted(rounds)}")
        # 外れ1位が同じ選手に重なることはありうるが、最終的な1位は球団ごとに異なる
        finals = [normalize_name(r.final.player_name) for r in self.first_round]
        if len(finals) != len(set(finals)):
            raise ValueError("最終的な1位に同じ選手が複数います")
        return self

    def by_team(self) -> dict[TeamCode, ActualFirstRound]:
        return {r.team: r for r in self.first_round}

    def lotte_acquired(self) -> list[ActualPlayer]:
        """ロッテが実際に獲得した1〜5位（1位は最終的な1位）。"""
        return [self.by_team()[TeamCode.M].final, *sorted(self.lotte, key=lambda p: p.round)]


def load_actual(path: Path) -> ActualDraft:
    return ActualDraft.model_validate_json(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "data/actual/2026.json")
    try:
        actual = load_actual(path)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    lost = [r.team.value for r in actual.first_round if not r.won_lottery]
    print(f"{path}: OK（抽選で外れた球団: {', '.join(lost) or 'なし'}、ロッテ2位以降 {len(actual.lotte)}人）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
