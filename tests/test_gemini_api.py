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
