"""予測出力の Pydantic スキーマ（docs/PLAN.md 3.5）。

LLM の出力は必ずこのスキーマで検証してから保存・採点する。
LLM には選手名だけを出力させ、候補者マスタ（player_id）との突合は採点側のコードで行う。
突合結果は予測ファイルに書き戻さない（事前登録後も突合ロジックを直せるようにするため）。
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


class Position(StrEnum):
    """NPB のドラフト結果と同じ4区分。"""

    PITCHER = "投手"
    CATCHER = "捕手"
    INFIELDER = "内野手"
    OUTFIELDER = "外野手"


class _Strict(BaseModel):
    # 未知のフィールドは弾く（LLM が余計なキーを返した場合も検出する）
    model_config = ConfigDict(extra="forbid")


def _name_key(player_name: str) -> str:
    """同一選手判定用のキー。空白（全角を含む）を除いた名前。"""
    return "".join(player_name.split())


class _PlayerRef(_Strict):
    """予測に出てくる選手。マスタとの突合は名前で行い、所属・ポジションは誤りの判定に使う。"""

    player_name: NonEmptyStr
    affiliation: NonEmptyStr  # 所属（高校・大学・企業・独立球団など）
    position: Position


class PlayerProb(_PlayerRef):
    prob: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class FirstRoundBid(_Strict):
    team: TeamCode
    candidates: list[PlayerProb] = Field(min_length=1)  # 入札候補と確率（合計 1.0）

    @model_validator(mode="after")
    def _check_candidates(self) -> "FirstRoundBid":
        total = sum(c.prob for c in self.candidates)
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"{self.team}: 確率の合計が 1.0 ではありません（{total:.4f}）")
        keys = [_name_key(c.player_name) for c in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.team}: 同じ選手が重複しています")
        return self


class ContestPrediction(_PlayerRef):
    """競合予測。PLAN.md に定義がないため最小限の初期案。"""

    num_teams: int = Field(ge=2, le=NUM_TEAMS)  # 競合する球団数


class LottePick(_PlayerRef):
    """ロッテの指名予測。1位は「入札する選手」を指す（抽選の結果や外れ1位ではない）。"""

    round: int = Field(ge=1, le=LOTTE_ROUNDS)
    reason: str
    sources: list[HttpUrl] = Field(default_factory=list)  # 条件 A/B では空になりうる


class PredictionOutput(_Strict):
    """LLM が出力する部分。LLM にはこのスキーマだけを渡す。"""

    first_round: list[FirstRoundBid]  # 12件
    contested: list[ContestPrediction] = Field(default_factory=list)  # 最小構成では対象外
    lotte: list[LottePick]  # 5件

    @model_validator(mode="after")
    def _check_first_round(self) -> "PredictionOutput":
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
    def _check_lotte(self) -> "PredictionOutput":
        rounds = sorted(p.round for p in self.lotte)
        if rounds != list(range(1, LOTTE_ROUNDS + 1)):
            raise ValueError(f"lotte は1〜{LOTTE_ROUNDS}位を1件ずつ含む必要があります（round={rounds}）")
        keys = [_name_key(p.player_name) for p in self.lotte]
        if len(keys) != len(set(keys)):
            raise ValueError("lotte で同じ選手が複数の順位に指名されています")
        return self


class Prediction(_Strict):
    """predictions/ に保存する1回分の予測。メタデータはコードが付与し、LLM には書かせない。"""

    run_id: NonEmptyStr
    model_id: NonEmptyStr
    condition: NonEmptyStr  # A / B / C / D / H-claude / ...
    output: PredictionOutput
