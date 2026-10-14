"""draft-db のツールと MCP サーバー。"""

import json

import anyio
import pytest
from mcp import Client

from src.candidates import Candidate
from src.draft_db_mcp.server import build_server
from src.draft_db_mcp.tools import SEARCH_LIMIT, TOOL_SPECS, DraftDB

URL = "https://example.com/list"


def _c(pid: str, name: str, cat: str, team: str, kana: str | None = None) -> Candidate:
    return Candidate(player_id=pid, name=name, kana=kana, category=cat, team=team, source_url=URL)


@pytest.fixture
def db() -> DraftDB:
    return DraftDB(
        [
            _c("HS-001", "織田 翔希", "高校", "横浜"),
            _c("HS-002", "髙橋 一郎", "高校", "横浜"),
            _c("UN-001", "渡部 海", "大学", "青山学院大", "わたべ かい"),
            _c("SH-001", "藤澤 涼介", "社会人", "東京ガス"),
        ]
    )


def _ids(res: dict) -> list[str]:
    return [p["player_id"] for p in res["players"]]


def test_search_by_name_ignores_spaces_and_variants(db: DraftDB) -> None:
    assert _ids(db.search_players("織田翔希")) == ["HS-001"]
    assert _ids(db.search_players("高橋")) == ["HS-002"]  # 髙 → 高


def test_search_by_kana_team_and_terms(db: DraftDB) -> None:
    assert _ids(db.search_players("わたべ")) == ["UN-001"]
    assert _ids(db.search_players("横浜")) == ["HS-001", "HS-002"]
    assert _ids(db.search_players("横浜 織田")) == ["HS-001"]  # すべての語を含む


def test_search_category_filter_and_empty_query(db: DraftDB) -> None:
    assert _ids(db.search_players("横浜", category="大学")) == []
    assert "error" in db.search_players("  ")


def test_search_truncates() -> None:
    many = DraftDB([_c(f"HS-{i:03d}", f"選手{i}", "高校", "某高") for i in range(1, SEARCH_LIMIT + 6)])
    res = many.search_players("某高")
    assert res["total"] == SEARCH_LIMIT + 5 and len(res["players"]) == SEARCH_LIMIT and res["truncated"]


def test_get_player_and_call_errors(db: DraftDB) -> None:
    assert json.loads(db.call("get_player", {"player_id": "SH-001"}))["team"] == "東京ガス"
    assert "error" in json.loads(db.call("get_player", {"player_id": "XX-999"}))
    assert "error" in json.loads(db.call("nope", {}))
    assert "error" in json.loads(db.call("search_players", {"q": "x"}))  # 引数名の誤り


def test_mcp_server_exposes_same_tools() -> None:
    async def main() -> None:
        async with Client(build_server()) as client:
            tools = (await client.list_tools()).tools
            assert [(t.name, t.description, t.input_schema) for t in tools] == [
                (s.name, s.description, s.parameters) for s in TOOL_SPECS
            ]
            res = await client.call_tool("search_players", {"query": "織田 翔希"})
            body = json.loads(res.content[0].text)
            assert body["players"][0]["name"] == "織田 翔希" and not res.is_error

    anyio.run(main)
