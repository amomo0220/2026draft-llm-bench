"""Gemini API。APIキーは GEMINI_API_KEY から読む。

条件 C・D では Google 検索（グラウンディング）を使う。条件 D では draft-db のツールも関数として渡し、呼び出しを往復する。
"""

import json
from typing import Any

from google import genai
from google.genai import types

from src.runners.base import MAX_TOOL_TURNS, TOOL_LIMIT_RESULT, RunnerResult, ToolBox


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


def _searches(resp: Any, grounding: Any) -> int:
    """検索語の数。関数呼び出しと併用したときは tool_call パートに、単独のときは grounding_metadata に入る。"""
    parts = resp.candidates[0].content.parts if resp.candidates and resp.candidates[0].content else None
    calls = [p.tool_call for p in parts or [] if p.tool_call]
    if calls:
        return sum(len((c.args or {}).get("queries") or [1]) for c in calls)
    return len(grounding.web_search_queries or []) if grounding else 0


def call(
    model_id: str,
    system: str,
    user: str,
    max_output_tokens: int,
    web_search: bool = False,
    tools: ToolBox | None = None,
) -> RunnerResult:
    client = genai.Client()
    tool_defs: list[types.Tool] = [types.Tool(google_search=types.GoogleSearch())] if web_search else []
    extra: dict[str, Any] = {}
    if tools:
        decls = [
            types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
            for t in tools.specs
        ]
        tool_defs.append(types.Tool(function_declarations=decls))
        # 関数呼び出しは自前で往復する。Google 検索と併用するには include_server_side_tool_invocations が必要
        extra["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        if web_search:
            extra["tool_config"] = types.ToolConfig(include_server_side_tool_invocations=True)
    if tool_defs:
        # ツールありでは JSON モードを使わず、プロンプトの指示と呼び出し側の検証で形式を守らせる（3社でそろえる）。
        # 検索回数を直接制限する設定はない
        extra["tools"] = tool_defs
    else:
        extra["response_mime_type"] = "application/json"
    # thinking は既定のまま。スキーマの検証は呼び出し側の Pydantic で行う
    config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_output_tokens, **extra)
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=user)])]
    in_tok = out_tok = thoughts_total = searches = functions = turns = max_in = 0

    while True:
        resp = client.models.generate_content(model=model_id, contents=contents, config=config)
        u = resp.usage_metadata
        if u:
            thoughts = u.thoughts_token_count or 0
            request_in = (u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0)
            in_tok += request_in
            max_in = max(max_in, request_in)
            out_tok += (u.candidates_token_count or 0) + thoughts  # 思考トークンも出力として課金される
            thoughts_total += thoughts
        cand = resp.candidates[0] if resp.candidates else None
        grounding = cand.grounding_metadata if cand else None
        searches += _searches(resp, grounding)
        fcs = (resp.function_calls or []) if tools else []
        if not fcs or cand is None or cand.content is None or turns > MAX_TOOL_TURNS:
            break
        over = turns == MAX_TOOL_TURNS
        turns += 1
        functions += 0 if over else len(fcs)
        contents.append(cand.content)  # thought_signature を含めてそのまま返す
        responses = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": json.loads(TOOL_LIMIT_RESULT if over else tools.call(fc.name or "", dict(fc.args or {})))},
                )
            )
            for fc in fcs
        ]
        contents.append(types.Content(role="user", parts=responses))

    text = resp.text or ""
    return RunnerResult(
        text=with_citations(text, grounding) if web_search else text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=thoughts_total,
        tool_calls=searches,
        function_calls=functions,
        max_request_input_tokens=max_in,
        stop_reason=str(cand.finish_reason) if cand and cand.finish_reason is not None else None,
    )
