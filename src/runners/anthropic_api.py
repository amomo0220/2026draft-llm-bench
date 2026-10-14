"""Anthropic Messages API。

APIキーは BENCH_ANTHROPIC_API_KEY から読む。ANTHROPIC_API_KEY はクラウド環境で Claude Code 自身の認証に
予約されていて使えないため。接続先も環境変数（ANTHROPIC_BASE_URL）に左右されないよう明示する。
条件 C では、サーバー側の Web 検索ツールを使う。
"""

import os
from typing import Any

import anthropic

from src.runners.base import MAX_SEARCHES, RunnerResult

API_KEY_ENV = "BENCH_ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES}
MAX_CONTINUATIONS = 5  # サーバー側の検索ループが pause_turn で止まったときに再開する上限


def _paused(msg: Any) -> bool:
    """サーバー側の検索ループの途中で止まったか。

    pause_turn のほか、クライアント側のツールを渡していないのに tool_use で止まり、最後のブロックが
    結果のないサーバー側ツールの呼び出しだった場合も、同じく送り返せば再開できるとみなす（試行で1回観測）。
    """
    if msg.stop_reason == "pause_turn":
        return True
    return msg.stop_reason == "tool_use" and bool(msg.content) and msg.content[-1].type == "server_tool_use"


def call(model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False) -> RunnerResult:
    client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    tools = [WEB_SEARCH_TOOL] if web_search else []
    in_tok = out_tok = searches = 0
    paused: list[Any] = []  # 一時停止した応答の内容（再開のたびに積み上げる）

    for _ in range(MAX_CONTINUATIONS + 1):
        # 出力が長いので、SDK のタイムアウトを避けるためストリーミングで受け取る。thinking・effort は既定のまま
        with client.messages.stream(
            model=model_id, max_tokens=max_output_tokens, system=system, messages=messages, tools=tools
        ) as stream:
            msg = stream.get_final_message()
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        if msg.usage.server_tool_use:
            searches += msg.usage.server_tool_use.web_search_requests or 0
        if not _paused(msg):
            break
        # 一時停止までの応答をすべて足して再送すると、サーバーが続きから再開する（"続けて" などは足さない）
        paused += msg.content
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": list(paused)}]

    # 引用付きの回答はテキストブロックが分割されるので連結する。最終の JSON は最後のテキストに入る
    text = "".join(b.text for b in msg.content if b.type == "text")
    stop = msg.stop_reason
    if stop != "end_turn" and msg.content:  # 想定外の停止の原因を追えるよう、最後のブロックの種類も残す
        stop = f"{stop}(last={msg.content[-1].type})"
    return RunnerResult(
        text=text, input_tokens=in_tok, output_tokens=out_tok, tool_calls=searches, stop_reason=stop
    )
