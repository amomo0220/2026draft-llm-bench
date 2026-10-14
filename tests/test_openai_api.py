"""OpenAI ランナーの関数呼び出しの往復（API は呼ばない）。"""

from types import SimpleNamespace as NS

from src.runners import openai_api


class FakeTools:
    specs = [NS(name="get_player", description="d", parameters={"type": "object"})]

    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return '{"ok": true}'


def _resp(rid, output, text="", input_tokens=100):
    usage = NS(input_tokens=input_tokens, output_tokens=10, output_tokens_details=NS(reasoning_tokens=4))
    return NS(id=rid, output=output, output_text=text, usage=usage, status="completed")


def test_function_calls_continue_from_previous_response(monkeypatch):
    replies = [
        _resp("r1", [NS(type="web_search_call"), NS(type="function_call", call_id="c1", name="get_player", arguments='{"player_id": "HS-001"}')]),
        _resp("r2", [NS(type="function_call", call_id="c2", name="get_player", arguments="not json")], input_tokens=150),
        _resp("r3", [NS(type="message")], "メモ"),
    ]
    requests = []

    def create(**kw):
        requests.append(kw)
        return replies.pop(0)

    monkeypatch.setattr(openai_api.openai, "OpenAI", lambda: NS(responses=NS(create=create)))
    tools = FakeTools()
    res = openai_api.call("m", "sys", "u", 1000, web_search=True, tools=tools)

    assert [t["type"] for t in requests[0]["tools"]] == ["web_search", "function"]
    assert "text" not in requests[0]  # ツールありでは JSON モードを使わない
    assert requests[1]["previous_response_id"] == "r1"
    assert requests[1]["input"] == [{"type": "function_call_output", "call_id": "c1", "output": '{"ok": true}'}]
    assert "error" in requests[2]["input"][0]["output"]  # 引数が JSON でなければツールを呼ばずにエラーを返す
    assert tools.calls == [("get_player", {"player_id": "HS-001"})]
    assert (res.text, res.tool_calls, res.function_calls) == ("メモ", 1, 2)
    assert (res.input_tokens, res.output_tokens, res.reasoning_tokens) == (350, 30, 12)
    assert res.max_request_input_tokens == 150  # 合計ではなく、1回のリクエストの最大


def test_json_mode_without_tools(monkeypatch):
    requests = []
    monkeypatch.setattr(
        openai_api.openai, "OpenAI", lambda: NS(responses=NS(create=lambda **kw: requests.append(kw) or _resp("r", [], "{}")))
    )
    openai_api.call("m", "sys", "u", 1000)
    assert requests[0]["text"] == {"format": {"type": "json_object"}} and "tools" not in requests[0]
