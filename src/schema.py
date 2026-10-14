"""予測出力の Pydantic スキーマ（docs/PLAN.md 3.5）。

LLM の出力は必ずこのスキーマで検証してから保存・採点する。
スキーマを変更するときは tests/test_schema.py も同時に更新すること。
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

# 確率の合計が 1.0 とみなす許容誤差。LLM が出す小数の丸め誤差を吸収する。
PROB_SUM_TOLERANCE = 1e-2

NUM_TEAMS = 12
LOTTE_ROUNDS = 5

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TeamCode(StrEnum):
    """NPB 12球団のコード。"""

    G = "G"  # 読売ジャイアンツ
    T = "T"  # 阪神タイガース
    DB = "DB"  # 横浜DeNAベイスターズ
    C = "C"  # 広島東洋カープ
    S = "S"  # 東京ヤクルトスワローズ
    D = "D"  # 中日ドラゴンズ
    H = "H"  # 福岡ソフトバンクホークス
    F = "F"  # 北海道日本ハムファイターズ
    B = "B"  # オリックス・バファローズ
    E = "E"  # 東北楽天ゴールデンイーグルス
    L = "L"  # 埼玉西武ライオンズ
    M = "M"  # 千葉ロッテマリーンズ


class _Strict(BaseModel):
    # 未知のフィールドは弾く（LLM が余計なキーを返した場合も検出する）
    model_config = ConfigDict(extra="forbid")


def _player_key(player_name: str, player_id: str | None) -> str:
    """同一選手判定用のキー。マスタ ID があれば優先し、なければ空白を除いた名前を使う。"""
    if player_id:
        return f"id:{player_id}"
    return "name:" + "".join(player_name.split())


class PlayerProb(_Strict):
    player_name: NonEmptyStr
    player_id: str | None = None  # マスタと突合できた場合
    prob: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class FirstRoundBid(_Strict):
    team: TeamCode
    candidates: list[PlayerProb] = Field(min_length=1)  # 入札候補と確率（合計 1.0）

    @model_validator(mode="after")
    def _check_candidates(self) -> "FirstRoundBid":
        total = sum(c.prob for c in self.candidates)
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"{self.team}: 確率の合計が 1.0 ではありません（{total:.4f}）")
        keys = [_player_key(c.player_name, c.player_id) for c in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.team}: 同じ選手が重複しています")
        return self


class ContestPrediction(_Strict):
    """競合予測。PLAN.md に定義がないため最小限の初期案。"""

    player_name: NonEmptyStr
    player_id: str | None = None
    num_teams: int = Field(ge=2, le=NUM_TEAMS)  # 競合する球団数


class LottePick(_Strict):
    round: int = Field(ge=1, le=LOTTE_ROUNDS)
    player_name: NonEmptyStr
    player_id: str | None = None
    reason: str
    sources: list[HttpUrl] = Field(default_factory=list)  # 条件 A/B では空になりうる


class Prediction(_Strict):
    run_id: NonEmptyStr
    model_id: NonEmptyStr
    condition: NonEmptyStr  # A / B / C / D / H-claude / ...
    first_round: list[FirstRoundBid]  # 12件
    contested: list[ContestPrediction] = Field(default_factory=list)  # 最小構成では対象外
    lotte: list[LottePick]  # 5件

    @model_validator(mode="after")
    def _check_first_round(self) -> "Prediction":
        teams = [b.team for b in self.first_round]
        if len(teams) != NUM_TEAMS or set(teams) != set(TeamCode):
            missing = sorted(set(TeamCode) - set(teams))
            dup = sorted({t for t in teams if teams.count(t) > 1})
            raise ValueError(
                f"first_round は12球団を1件ずつ含む必要があります"
                f"（件数={len(teams)}, 不足={missing}, 重複={dup}）"
            )
        return self

    @model_validator(mode="after")
    def _check_lotte(self) -> "Prediction":
        rounds = sorted(p.round for p in self.lotte)
        if rounds != list(range(1, LOTTE_ROUNDS + 1)):
            raise ValueError(f"lotte は1〜{LOTTE_ROUNDS}位を1件ずつ含む必要があります（round={rounds}）")
        keys = [_player_key(p.player_name, p.player_id) for p in self.lotte]
        if len(keys) != len(set(keys)):
            raise ValueError("lotte で同じ選手が複数の順位に指名されています")
        return self
