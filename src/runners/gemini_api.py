"""Gemini API。APIキーは GEMINI_API_KEY から読む。条件 C では Google 検索（グラウンディング）を使う。"""

from google import genai
from google.genai import types

from src.runners.base import RunnerResult


def call(model_id: str, system: str, user: str, max_output_tokens: int, web_search: bool = False) -> RunnerResult:
    client = genai.Client()
    if web_search:
        # 検索ありでは JSON モードを使わず、プロンプトの指示と呼び出し側の検証で形式を守らせる（3社でそろえる）。
        # 検索回数を直接制限する設定はない
        extra = {"tools": [types.Tool(google_search=types.GoogleSearch())]}
    else:
        extra = {"response_mime_type": "application/json"}
    # thinking は既定のまま。スキーマの検証は呼び出し側の Pydantic で行う
    resp = client.models.generate_content(
        model=model_id,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_output_tokens, **extra),
    )
    u = resp.usage_metadata
    thoughts = (u.thoughts_token_count or 0) if u else 0
    candidates = (u.candidates_token_count or 0) if u else 0
    cand = resp.candidates[0] if resp.candidates else None
    grounding = cand.grounding_metadata if cand else None
    searches = len(grounding.web_search_queries or []) if grounding else 0
    return RunnerResult(
        text=resp.text or "",
        input_tokens=(u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0) if u else 0,
        output_tokens=candidates + thoughts,  # 思考トークンも出力として課金される
        reasoning_tokens=thoughts,
        tool_calls=searches,
        stop_reason=str(cand.finish_reason) if cand and cand.finish_reason is not None else None,
    )
