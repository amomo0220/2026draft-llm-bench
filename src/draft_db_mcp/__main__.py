"""draft-db の MCP サーバーを stdio で起動する。

使い方:
    uv run python -m src.draft_db_mcp
"""

import anyio
from mcp.server.stdio import stdio_server

from src.draft_db_mcp.server import build_server


async def main() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
