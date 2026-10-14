"""OpenAI Responses API（条件 A・B 用。検索なし）。APIキーは OPENAI_API_KEY から読む。"""

import openai

from src.runners.base import RunnerResult


def call(model_id: str, system: str, user: str, max_output_tokens: int) -> RunnerResult:
    client = openai.OpenAI()
    # reasoning は既定のまま。JSON モードで出力させ、スキーマの検証は呼び出し側の Pydantic で行う
    resp = client.responses.create(
        model=model_id,
        instructions=system,
        input=user,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
    )
    usage = resp.usage
    reasoning = usage.output_tokens_details.reasoning_tokens if usage and usage.output_tokens_details else None
    return RunnerResult(
        text=resp.output_text,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        reasoning_tokens=reasoning,
        stop_reason=resp.status,
    )
