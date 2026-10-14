"""Gemini の検索の情報源を調査メモに書き込む処理（API は呼ばない）。"""

from types import SimpleNamespace as NS

from src.runners.gemini_api import with_citations


def _chunk(title, uri):
    return NS(web=NS(title=title, uri=uri))


def _support(start, end, idx):
    return NS(segment=NS(start_index=start, end_index=end), grounding_chunk_indices=idx)


def test_inserts_markers_at_byte_offsets_and_lists_sources() -> None:
    text = "阪神は渡部。広島は未定。"
    b = text.encode()
    end1 = len("阪神は渡部。".encode())  # 日本語は1文字3バイト
    g = NS(
        grounding_chunks=[_chunk("a.jp", "https://r/1"), _chunk("b.jp", "https://r/2")],
        grounding_supports=[_support(None, end1, [0, 1]), _support(end1, len(b), [1])],
    )
    out = with_citations(text, g)
    assert out.startswith("阪神は渡部。[1][2]広島は未定。[2]")
    assert "[1] a.jp https://r/1" in out and "[2] b.jp https://r/2" in out


def test_without_grounding_returns_text() -> None:
    assert with_citations("x", None) == "x"
    assert with_citations("x", NS(grounding_chunks=None, grounding_supports=None)) == "x"


def test_offset_inside_character_moves_to_boundary() -> None:
    g = NS(grounding_chunks=[_chunk("a.jp", "https://r/1")], grounding_supports=[_support(0, 4, [0])])
    assert with_citations("阪神は", g).startswith("阪神[1]は")  # 4バイト目は「神」の途中


def test_function_calls_round_trip(monkeypatch) -> None:
    from src.runners import gemini_api

    class Tools:
        specs = [NS(name="search_players", description="d", parameters={"type": "object"})]
        calls: list = []

        def call(self, name, args):
            self.calls.append((name, args))
            return '{"total": 1}'

    def resp(parts, fcs, text):
        cand = NS(content=NS(parts=parts), grounding_metadata=None, finish_reason="STOP")
        usage = NS(thoughts_token_count=5, prompt_token_count=100, tool_use_prompt_token_count=20, candidates_token_count=10)
        return NS(candidates=[cand], usage_metadata=usage, function_calls=fcs, text=text)

    fc = NS(id="f1", name="search_players", args={"query": "織田"})
    search = NS(tool_call=NS(args={"queries": ["a", "b"]}))
    replies = [resp([search, NS(tool_call=None)], [fc], None), resp([NS(tool_call=None)], [], "メモ")]
    seen = []

    def generate_content(model, contents, config):
        seen.append((list(contents), config))
        return replies.pop(0)

    monkeypatch.setattr(gemini_api.genai, "Client", lambda: NS(models=NS(generate_content=generate_content)))
    tools = Tools()
    res = gemini_api.call("m", "sys", "u", 1000, web_search=True, tools=tools)

    config = seen[0][1]
    assert config.tool_config.include_server_side_tool_invocations and config.response_mime_type is None
    assert tools.calls == [("search_players", {"query": "織田"})]
    sent = seen[1][0]
    assert len(sent) == 3 and sent[2].parts[0].function_response.response == {"result": {"total": 1}}
    assert sent[2].parts[0].function_response.id == "f1"
    assert (res.text, res.tool_calls, res.function_calls) == ("メモ", 2, 1)
    assert (res.input_tokens, res.output_tokens, res.reasoning_tokens) == (240, 30, 10)
