"""Gemini API（条件 A・B 用。検索なし）。APIキーは GEMINI_API_KEY から読む。"""

from google import genai
from google.genai import types

from src.runners.base import RunnerResult


def call(model_id: str, system: str, user: str, max_output_tokens: int) -> RunnerResult:
    client = genai.Client()
    # thinking は既定のまま。JSON モードで出力させ、スキーマの検証は呼び出し側の Pydantic で行う
    resp = client.models.generate_content(
        model=model_id,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    u = resp.usage_metadata
    thoughts = (u.thoughts_token_count or 0) if u else 0
    candidates = (u.candidates_token_count or 0) if u else 0
    finish = resp.candidates[0].finish_reason if resp.candidates else None
    return RunnerResult(
        text=resp.text or "",
        input_tokens=(u.prompt_token_count or 0) if u else 0,
        output_tokens=candidates + thoughts,  # 思考トークンも出力として課金される
        reasoning_tokens=thoughts,
        stop_reason=str(finish) if finish is not None else None,
    )
