"""OpenAI Responses API。APIキーは OPENAI_API_KEY から読む。条件 C では Web 検索ツールを使う。"""

from typing import Any

import openai

from src.runners.base import MAX_SEARCHES, RunnerResult


def call(model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False) -> RunnerResult:
    client = openai.OpenAI()
    kwargs: dict[str, Any] = {}
    if web_search:
        # 検索ありでは JSON モードを使わず、プロンプトの指示と呼び出し側の検証で形式を守らせる（3社でそろえる）
        kwargs = {"tools": [{"type": "web_search"}], "max_tool_calls": MAX_SEARCHES}
    else:
        kwargs = {"text": {"format": {"type": "json_object"}}}
    # reasoning は既定のまま。スキーマの検証は呼び出し側の Pydantic で行う
    resp = client.responses.create(
        model=model_id, instructions=system, input=user, max_output_tokens=max_output_tokens, **kwargs
    )
    usage = resp.usage
    reasoning = usage.output_tokens_details.reasoning_tokens if usage and usage.output_tokens_details else None
    searches = sum(1 for item in resp.output if item.type == "web_search_call")
    return RunnerResult(
        text=resp.output_text,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        reasoning_tokens=reasoning,
        tool_calls=searches,
        stop_reason=resp.status,
    )
