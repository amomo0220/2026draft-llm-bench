"""Anthropic ランナーの pause_turn からの再開（API は呼ばない）。"""

from types import SimpleNamespace

import pytest

from src.runners import anthropic_api


def _msg(blocks, stop_reason, searches):
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=10, server_tool_use=SimpleNamespace(web_search_requests=searches)
    )
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=usage)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _search(i):
    return SimpleNamespace(type="server_tool_use", id=f"s{i}")


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)

        class _Ctx:
            def __enter__(self_inner):
                return SimpleNamespace(get_final_message=lambda: reply)

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


@pytest.fixture
def fake(monkeypatch):
    def install(replies):
        client = FakeClient(replies)
        monkeypatch.setenv(anthropic_api.API_KEY_ENV, "test")
        monkeypatch.setattr(anthropic_api.anthropic, "Anthropic", lambda **kw: client)
        return client

    return install


def test_pause_turn_resumes_with_all_paused_content(fake):
    b1, b2 = _search(1), _search(2)
    client = fake([_msg([b1], "pause_turn", 5), _msg([b2], "pause_turn", 4), _msg([_text('{"a"'), _text(": 1}")], "end_turn", 1)])

    res = anthropic_api.call("m", "sys", "u", 1000, web_search=True)

    assert len(client.requests) == 3
    assert client.requests[0]["messages"] == [{"role": "user", "content": "u"}]
    assert client.requests[1]["messages"][1] == {"role": "assistant", "content": [b1]}
    # 2回目の再開では、1回目と2回目の一時停止までの内容をどちらも送り返す
    assert client.requests[2]["messages"][1] == {"role": "assistant", "content": [b1, b2]}
    assert res.text == '{"a": 1}'
    assert (res.input_tokens, res.output_tokens, res.tool_calls) == (300, 30, 10)
    assert res.stop_reason == "end_turn"


def test_no_tools_without_web_search(fake):
    client = fake([_msg([_text("{}")], "end_turn", 0)])
    anthropic_api.call("m", "sys", "u", 1000)
    assert client.requests[0]["tools"] == []


def test_tool_use_stop_on_server_tool_resumes(fake):
    b1 = _search(1)
    client = fake([_msg([b1], "tool_use", 1), _msg([_text("{}")], "end_turn", 1)])
    res = anthropic_api.call("m", "sys", "u", 1000, web_search=True)
    assert len(client.requests) == 2 and client.requests[1]["messages"][1]["content"] == [b1]
    assert res.text == "{}" and res.tool_calls == 2


def test_tool_use_stop_on_other_block_does_not_resume(fake):
    client = fake([_msg([_text("メモ")], "tool_use", 1)])
    res = anthropic_api.call("m", "sys", "u", 1000, web_search=True)
    assert len(client.requests) == 1 and res.stop_reason == "tool_use(last=text)"


class FakeTools:
    specs = [SimpleNamespace(name="search_players", description="d", parameters={"type": "object"})]

    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return '{"total": 0}'


def _use(i):
    return SimpleNamespace(type="tool_use", id=f"t{i}", name="search_players", input={"query": f"q{i}"})


def test_function_tools_round_trip(fake):
    u1, u2 = _use(1), _use(2)
    client = fake([_msg([_text("確認します"), u1, u2], "tool_use", 0), _msg([_text("メモ")], "end_turn", 2)])
    tools = FakeTools()

    res = anthropic_api.call("m", "sys", "u", 1000, web_search=True, tools=tools)

    assert [t["name"] for t in client.requests[0]["tools"]] == ["web_search", "search_players"]
    assert tools.calls == [("search_players", {"query": "q1"}), ("search_players", {"query": "q2"})]
    sent = client.requests[1]["messages"]
    assert sent[1]["role"] == "assistant" and sent[1]["content"][1:] == [u1, u2]
    assert sent[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": '{"total": 0}'},
            {"type": "tool_result", "tool_use_id": "t2", "content": '{"total": 0}'},
        ],
    }
    assert res.text == "メモ" and res.function_calls == 2 and res.tool_calls == 2


def test_function_tool_turns_are_capped(fake, monkeypatch):
    monkeypatch.setattr(anthropic_api, "MAX_TOOL_TURNS", 2)
    client = fake([_msg([_use(i)], "tool_use", 0) for i in range(4)])
    tools = FakeTools()
    res = anthropic_api.call("m", "sys", "u", 1000, tools=tools)
    # 上限の後の呼び出しは実行せず、上限に達したと返してもう1回だけ応答させる
    assert len(client.requests) == 4 and len(tools.calls) == 2 and res.function_calls == 2
    assert "上限" in client.requests[3]["messages"][-1]["content"][0]["content"]
    assert res.stop_reason == "tool_use(last=tool_use)"


def test_limit_lets_model_answer(fake, monkeypatch):
    monkeypatch.setattr(anthropic_api, "MAX_TOOL_TURNS", 1)
    fake([_msg([_use(1)], "tool_use", 0), _msg([_use(2)], "tool_use", 0), _msg([_text("メモ")], "end_turn", 0)])
    res = anthropic_api.call("m", "sys", "u", 1000, tools=FakeTools())
    assert res.text == "メモ" and res.function_calls == 1
