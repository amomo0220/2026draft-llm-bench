"""Anthropic Messages API。

APIキーは BENCH_ANTHROPIC_API_KEY から読む。ANTHROPIC_API_KEY はクラウド環境で Claude Code 自身の認証に
予約されていて使えないため。接続先も環境変数（ANTHROPIC_BASE_URL）に左右されないよう明示する。
条件 C・D では、サーバー側の Web 検索ツールを使う。条件 D では draft-db のツールも関数として渡し、呼び出しを往復する。
"""

import os
from typing import Any

import anthropic

from src.runners.base import MAX_SEARCHES, MAX_TOOL_TURNS, TOOL_LIMIT_RESULT, RunnerResult, ToolBox

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


def _append_assistant(messages: list[dict[str, Any]], content: list[Any]) -> None:
    """応答をアシスタントの発言として足す。直前もアシスタント（一時停止した応答）ならつなげる。"""
    if messages[-1]["role"] == "assistant":
        messages[-1] = {"role": "assistant", "content": messages[-1]["content"] + list(content)}
    else:
        messages.append({"role": "assistant", "content": list(content)})


def call(
    model_id: str,
    system: str,
    user: str,
    max_output_tokens: int,
    web_search: bool = False,
    tools: ToolBox | None = None,
) -> RunnerResult:
    client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    tool_defs: list[dict[str, Any]] = [WEB_SEARCH_TOOL] if web_search else []
    if tools:
        tool_defs += [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools.specs]
    in_tok = out_tok = searches = functions = max_in = 0
    continuations = turns = 0

    while True:
        # 出力が長いので、SDK のタイムアウトを避けるためストリーミングで受け取る。thinking・effort は既定のまま
        with client.messages.stream(
            model=model_id, max_tokens=max_output_tokens, system=system, messages=list(messages), tools=tool_defs
        ) as stream:
            msg = stream.get_final_message()
        in_tok += msg.usage.input_tokens
        max_in = max(max_in, msg.usage.input_tokens)
        out_tok += msg.usage.output_tokens
        if msg.usage.server_tool_use:
            searches += msg.usage.server_tool_use.web_search_requests or 0
        uses = [b for b in msg.content if b.type == "tool_use"] if tools else []
        if msg.stop_reason == "tool_use" and uses and turns <= MAX_TOOL_TURNS:
            over = turns == MAX_TOOL_TURNS
            turns += 1
            functions += 0 if over else len(uses)
            _append_assistant(messages, msg.content)
            results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": TOOL_LIMIT_RESULT if over else tools.call(b.name, dict(b.input))}
                for b in uses
            ]
            messages.append({"role": "user", "content": results})
        elif _paused(msg) and continuations < MAX_CONTINUATIONS:
            # 一時停止までの応答をすべて足して再送すると、サーバーが続きから再開する（"続けて" などは足さない）
            continuations += 1
            _append_assistant(messages, msg.content)
        else:
            break

    # 引用付きの回答はテキストブロックが分割されるので連結する。最終の JSON は最後のテキストに入る
    text = "".join(b.text for b in msg.content if b.type == "text")
    stop = msg.stop_reason
    if stop != "end_turn" and msg.content:  # 想定外の停止の原因を追えるよう、最後のブロックの種類も残す
        stop = f"{stop}(last={msg.content[-1].type})"
    return RunnerResult(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        tool_calls=searches,
        function_calls=functions,
        max_request_input_tokens=max_in,
        stop_reason=stop,
    )
