"""Gemini API。APIキーは GEMINI_API_KEY から読む。条件 C では Google 検索（グラウンディング）を使う。"""

from typing import Any

from google import genai
from google.genai import types

from src.runners.base import RunnerResult


def with_citations(text: str, grounding: Any) -> str:
    """検索の情報源を調査メモに書き込む。

    Gemini は情報源の URL を本文に書かず grounding_metadata で返す。他社とそろえるため、根拠のある文の後に [n] を差し込み、
    末尾に情報源の一覧を付ける。URL は Google のリダイレクト用のもの（vertexaisearch.cloud.google.com）で、title はドメイン名。
    segment の位置は UTF-8 のバイト数で数える。
    """
    chunks = (grounding.grounding_chunks or []) if grounding else []
    if not chunks:
        return text
    data = text.encode("utf-8")
    marks: dict[int, list[int]] = {}
    for sup in grounding.grounding_supports or []:
        if sup.segment and sup.segment.end_index is not None and sup.grounding_chunk_indices:
            marks.setdefault(sup.segment.end_index, []).extend(sup.grounding_chunk_indices)
    for end in sorted(marks, reverse=True):  # 後ろから差し込むと、前の位置がずれない
        tag = "".join(f"[{i + 1}]" for i in sorted(set(marks[end])))
        pos = min(end, len(data))
        while pos < len(data) and data[pos] & 0xC0 == 0x80:  # 文字の途中なら次の文字の境界までずらす
            pos += 1
        data = data[:pos] + tag.encode("utf-8") + data[pos:]
    sources = [f"[{i + 1}] {c.web.title or ''} {c.web.uri}" for i, c in enumerate(chunks) if c.web and c.web.uri]
    return data.decode("utf-8") + "\n\n## 情報源\n\n" + "\n".join(sources) + "\n"


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
        text=with_citations(resp.text or "", grounding) if web_search else resp.text or "",
        input_tokens=(u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0) if u else 0,
        output_tokens=candidates + thoughts,  # 思考トークンも出力として課金される
        reasoning_tokens=thoughts,
        tool_calls=searches,
        stop_reason=str(cand.finish_reason) if cand and cand.finish_reason is not None else None,
    )
