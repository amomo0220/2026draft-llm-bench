"""予測に出てくる選手名を候補者マスタと突合する（docs/PLAN.md 3.4, 3.5）。

採点ロジックの一部なので、事前登録後も修正してよい（予測ファイルは変更しない）。
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from src.candidates import Candidate
from src.schema import Position

# 異体字・旧字体を、突合用に常用字体へそろえる。
# 公式一覧でも表示できない字があり（例: 𠮷）、LLM は常用字体で答えることが多いため。
_VARIANTS = str.maketrans(
    {
        "𠮷": "吉",
        "髙": "高",
        "﨑": "崎",
        "嵜": "崎",
        "邊": "辺",
        "邉": "辺",
        "澤": "沢",
        "濱": "浜",
        "廣": "広",
        "齋": "斎",
        "齊": "斉",
        "櫻": "桜",
        "嶋": "島",
        "德": "徳",
        "槇": "槙",
        "國": "国",
        "瀨": "瀬",
        "眞": "真",
        "冨": "富",
        "條": "条",
        "龍": "竜",
        "惠": "恵",
        "實": "実",
        "鐵": "鉄",
        "昻": "昂",
        "禮": "礼",
        "﨔": "欅",
    }
)
_SEPARATORS = re.compile(r"[\s・･]+")


def normalize_name(name: str) -> str:
    """突合用の名前キー。NFKC 正規化、異体字の統一、空白・中黒の除去。"""
    return _SEPARATORS.sub("", unicodedata.normalize("NFKC", name).translate(_VARIANTS))


_AFFILIATION_SUFFIX = re.compile(r"(高等学校|高校|高|大学|大)$")


def normalize_affiliation(affiliation: str) -> str:
    """突合用の所属キー。「横浜高」と「横浜」、「青山学院大」と「青山学院大学」を同じとみなす。"""
    key = normalize_name(affiliation)
    return _AFFILIATION_SUFFIX.sub("", key)


def same_affiliation(a: str, b: str) -> bool:
    return normalize_affiliation(a) == normalize_affiliation(b)


_POSITION_WORDS: dict[str, Position] = {
    "投手": Position.PITCHER,
    "捕手": Position.CATCHER,
    "内野手": Position.INFIELDER,
    "一塁手": Position.INFIELDER,
    "二塁手": Position.INFIELDER,
    "三塁手": Position.INFIELDER,
    "遊撃手": Position.INFIELDER,
    "外野手": Position.OUTFIELDER,
    "左翼手": Position.OUTFIELDER,
    "中堅手": Position.OUTFIELDER,
    "右翼手": Position.OUTFIELDER,
}


def to_positions(text: str | None) -> set[Position]:
    """マスタのポジション表記（例: 「遊撃手」「投手、内野手」）を4区分の集合にする。不明な語は無視する。"""
    if not text:
        return set()
    words = re.split(r"[、,/／・\s]+", text)
    return {_POSITION_WORDS[w] for w in words if w in _POSITION_WORDS}


class MatchStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"  # 同姓同名が複数いて所属でも絞れない
    UNMATCHED = "unmatched"  # マスタにない（要確認）


@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    candidate: Candidate | None = None


class Master:
    """候補者マスタの名前索引。"""

    def __init__(self, candidates: list[Candidate]) -> None:
        self._by_name: dict[str, list[Candidate]] = {}
        for c in candidates:
            self._by_name.setdefault(normalize_name(c.name), []).append(c)

    def match(self, player_name: str, affiliation: str | None = None) -> MatchResult:
        hits = self._by_name.get(normalize_name(player_name), [])
        if len(hits) > 1 and affiliation is not None:
            hits = [c for c in hits if same_affiliation(c.team, affiliation)] or hits
        if not hits:
            return MatchResult(MatchStatus.UNMATCHED)
        if len(hits) > 1:
            return MatchResult(MatchStatus.AMBIGUOUS)
        return MatchResult(MatchStatus.MATCHED, hits[0])
