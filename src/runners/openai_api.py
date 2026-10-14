"""OpenAI Responses API。APIキーは OPENAI_API_KEY から読む。

条件 C・D では Web 検索ツールを使う。条件 D では draft-db のツールも関数として渡し、呼び出しを往復する。
"""

import json
from typing import Any

import openai

from src.runners.base import MAX_SEARCHES, MAX_TOOL_TURNS, TOOL_LIMIT_RESULT, RunnerResult, ToolBox


def call(
    model_id: str,
    system: str,
    user: str,
    max_output_tokens: int,
    web_search: bool = False,
    tools: ToolBox | None = None,
) -> RunnerResult:
    client = openai.OpenAI()
    tool_defs: list[dict[str, Any]] = [{"type": "web_search"}] if web_search else []
    if tools:
        tool_defs += [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools.specs
        ]
    kwargs: dict[str, Any] = {}
    if tool_defs:
        # ツールありでは JSON モードを使わず、プロンプトの指示と呼び出し側の検証で形式を守らせる（3社でそろえる）
        kwargs = {"tools": tool_defs, "max_tool_calls": MAX_SEARCHES} if web_search else {"tools": tool_defs}
    else:
        kwargs = {"text": {"format": {"type": "json_object"}}}

    # reasoning は既定のまま。スキーマの検証は呼び出し側の Pydantic で行う
    resp = client.responses.create(
        model=model_id, instructions=system, input=user, max_output_tokens=max_output_tokens, **kwargs
    )
    in_tok = out_tok = searches = functions = turns = 0
    reasoning: int | None = None
    while True:
        usage = resp.usage
        if usage:
            in_tok += usage.input_tokens
            out_tok += usage.output_tokens
            if usage.output_tokens_details:
                reasoning = (reasoning or 0) + usage.output_tokens_details.reasoning_tokens
        searches += sum(1 for item in resp.output if item.type == "web_search_call")
        fcs = [item for item in resp.output if item.type == "function_call"] if tools else []
        if not fcs or turns > MAX_TOOL_TURNS:
            break
        over = turns == MAX_TOOL_TURNS
        turns += 1
        functions += 0 if over else len(fcs)
        outputs = [
            {
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": TOOL_LIMIT_RESULT if over else _run(tools, fc.name, fc.arguments),
            }
            for fc in fcs
        ]
        # 前の応答（推論・検索の結果を含む）はサーバー側に残っているので、previous_response_id で続ける
        resp = client.responses.create(
            model=model_id,
            instructions=system,
            previous_response_id=resp.id,
            input=outputs,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    return RunnerResult(
        text=resp.output_text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=reasoning,
        tool_calls=searches,
        function_calls=functions,
        stop_reason=resp.status,
    )


def _run(tools: ToolBox, name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"引数が JSON ではありません: {e}"}, ensure_ascii=False)
    return tools.call(name, args if isinstance(args, dict) else {})
