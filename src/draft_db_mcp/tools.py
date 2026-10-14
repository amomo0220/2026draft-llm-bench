"""draft-db のツール（候補者マスタの検索）。MCP サーバー（条件 H）と API の関数呼び出し（条件 D）で共通に使う。

ツールの名前・説明・引数はここだけで定義する。docs/PLAN.md 3.2 条件 D。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.candidates import Candidate, Category, load_candidates
from src.matching import normalize_name

DEFAULT_CANDIDATES = Path("data/candidates.csv")
SEARCH_LIMIT = 30

COVERAGE = (
    "高校生・大学生は、プロ野球志望届の提出者（NPBドラフト対象者）を公式の一覧から全員収録している。"
    "ここにいない高校生・大学生は指名できない。社会人・独立リーグ・二軍参加球団の選手は主な候補だけで、網羅していない。"
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema（object）


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_players",
        description=(
            "2026年ドラフトの候補者マスタを、氏名・ふりがな・所属の一部で検索する。"
            "空白区切りで複数の語を渡すと、すべての語を含む選手を返す（例: 「横浜 織田」）。"
            "異体字（髙・﨑など）と空白の違いは無視する。" + COVERAGE
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "氏名・ふりがな・所属の一部"},
                "category": {
                    "type": "string",
                    "enum": [c.value for c in Category],
                    "description": "区分で絞り込む（省略可）",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_player",
        description="player_id（search_players の結果にある）で、候補者1人の全項目（投打・出典URLを含む）を返す。",
        parameters={
            "type": "object",
            "properties": {"player_id": {"type": "string", "description": "例: HS-001"}},
            "required": ["player_id"],
        },
    ),
]


def _summary(c: Candidate) -> dict[str, Any]:
    return {
        "player_id": c.player_id,
        "name": c.name,
        "kana": c.kana,
        "category": c.category.value,
        "team": c.team,
        "position": c.position,
    }


class DraftDB:
    def __init__(self, candidates: list[Candidate]) -> None:
        self._rows = candidates
        self._by_id = {c.player_id: c for c in candidates}
        self._keys = {c.player_id: [normalize_name(x) for x in (c.name, c.kana or "", c.team) if x] for c in candidates}

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    @classmethod
    def load(cls, path: Path = DEFAULT_CANDIDATES) -> "DraftDB":
        return cls(load_candidates(path))

    def search_players(self, query: str, category: str | None = None) -> dict[str, Any]:
        whole = normalize_name(query)
        if not whole:
            return {"error": "query が空です"}
        terms = [normalize_name(t) for t in query.split()] or [whole]
        hits = []
        for c in self._rows:
            if category and c.category.value != category:
                continue
            keys = self._keys[c.player_id]
            if (whole and any(whole in k for k in keys)) or all(any(t in k for k in keys) for t in terms if t):
                hits.append(c)
        return {"total": len(hits), "players": [_summary(c) for c in hits[:SEARCH_LIMIT]], "truncated": len(hits) > SEARCH_LIMIT}

    def get_player(self, player_id: str) -> dict[str, Any]:
        c = self._by_id.get(player_id.strip())
        if c is None:
            return {"error": f"player_id が見つかりません: {player_id}"}
        return c.model_dump(mode="json")

    def call(self, name: str, args: dict[str, Any]) -> str:
        """ツールを実行して JSON 文字列を返す。引数の誤りもエラーとして返し、モデルに直させる。"""
        fns: dict[str, Callable[..., dict[str, Any]]] = {"search_players": self.search_players, "get_player": self.get_player}
        try:
            result = fns[name](**args)
        except KeyError:
            result = {"error": f"未知のツールです: {name}"}
        except TypeError as e:
            result = {"error": f"引数が不正です: {e}"}
        return json.dumps(result, ensure_ascii=False)
