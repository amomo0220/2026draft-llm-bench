"""draft-db の MCP サーバー。ツールの定義と中身は tools.py と共通。"""

from pathlib import Path
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server

from src.draft_db_mcp.tools import DEFAULT_CANDIDATES, TOOL_SPECS, DraftDB


def build_server(candidates: Path = DEFAULT_CANDIDATES) -> Server[Any]:
    db = DraftDB.load(candidates)

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[types.Tool(name=t.name, description=t.description, input_schema=t.parameters) for t in TOOL_SPECS]
        )

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        text = db.call(params.name, dict(params.arguments or {}))
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=text.startswith('{"error"'))

    return Server("draft-db", on_list_tools=list_tools, on_call_tool=call_tool)
