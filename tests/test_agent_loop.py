"""Agent-loop tests using a scripted fake client: no API calls, fully deterministic."""

from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest

from src.vta.agent import loop

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def response(blocks, stop):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """Returns scripted responses in order; an Exception in the script is raised."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(loop, "client", client)
        return client

    return install


def tool_results_sent(client):
    """The tool_result blocks the loop sent back to the model."""
    messages = client.calls[0]["messages"]  # same list object the loop keeps appending to
    return next(m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], list))


def test_direct_answer(fake):
    fake([response([text("hello")], "end_turn")])
    r = loop.run_agent("hi", tool_functions={})
    assert (r["status"], r["answer"], r["rounds"]) == ("ok", "hello", 1)
    assert r["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_tool_exception_becomes_is_error_result_and_loop_continues(fake):
    def boom(**_):
        raise RuntimeError("outage")

    client = fake([response([tool_use("t1", "boom")], "tool_use"), response([text("sorry")], "end_turn")])
    r = loop.run_agent("q", tool_functions={"boom": boom}, tools=[])
    assert r["status"] == "ok" and r["trace"][0]["is_error"] is True
    sent = tool_results_sent(client)[0]
    assert sent["is_error"] is True and "RuntimeError: outage" in sent["content"]


def test_unknown_tool_is_reported_to_the_model(fake):
    client = fake([response([tool_use("t1", "nope")], "tool_use"), response([text("ok")], "end_turn")])
    loop.run_agent("q", tool_functions={"real": lambda: {}}, tools=[])
    sent = tool_results_sent(client)[0]
    assert sent["is_error"] and "Unknown tool 'nope'" in sent["content"] and "real" in sent["content"]


def test_bad_arguments_are_reported_not_raised(fake):
    client = fake([response([tool_use("t1", "f", wrong=1)], "tool_use"), response([text("ok")], "end_turn")])
    r = loop.run_agent("q", tool_functions={"f": lambda symbol: {"found": True}}, tools=[])
    assert r["status"] == "ok" and tool_results_sent(client)[0]["is_error"]


def test_not_found_is_a_normal_result_not_an_error(fake):
    client = fake([response([tool_use("t1", "f")], "tool_use"), response([text("none")], "end_turn")])
    loop.run_agent("q", tool_functions={"f": lambda: {"found": False, "error": "no such gene"}}, tools=[])
    assert "is_error" not in tool_results_sent(client)[0]


@pytest.mark.parametrize(
    "stop, status",
    [("max_tokens", "truncated"), ("refusal", "refusal"), ("something_new", "unexpected_stop")],
)
def test_non_normal_stop_reasons_are_surfaced(fake, stop, status):
    fake([response([text("partial")], stop)])
    r = loop.run_agent("q", tool_functions={})
    assert r["status"] == status and r["error"]


def test_iteration_cap(fake):
    fake([response([tool_use(f"t{i}", "f", n=i)], "tool_use") for i in range(5)])
    r = loop.run_agent("q", max_iterations=2, tool_functions={"f": lambda n: {"found": True}}, tools=[])
    assert r["status"] == "max_iterations" and r["rounds"] == 2 and len(r["trace"]) == 2


def test_identical_successful_call_is_not_re_executed(fake):
    calls = []

    def f(x):
        calls.append(x)
        return {"found": True}

    fake([response([tool_use("a", "f", x=1)], "tool_use"), response([tool_use("b", "f", x=1)], "tool_use"), response([text("done")], "end_turn")])
    r = loop.run_agent("q", tool_functions={"f": f}, tools=[])
    assert calls == [1]
    assert r["trace"][1]["is_error"] is True


def test_identical_call_after_not_found_is_allowed_to_retry(fake):
    calls = []

    def f(x):
        calls.append(x)
        return {"found": False, "error": "request failed"}

    fake([response([tool_use("a", "f", x=1)], "tool_use"), response([tool_use("b", "f", x=1)], "tool_use"), response([text("done")], "end_turn")])
    loop.run_agent("q", tool_functions={"f": f}, tools=[])
    assert calls == [1, 1]


def test_oversized_tool_result_is_truncated(fake):
    client = fake([response([tool_use("a", "f")], "tool_use"), response([text("ok")], "end_turn")])
    loop.run_agent("q", tool_functions={"f": lambda: {"blob": "x" * 50_000}}, tools=[])
    content = tool_results_sent(client)[0]["content"]
    assert len(content) < loop.MAX_TOOL_RESULT_CHARS + 100 and content.endswith("size limit]")


def test_rejected_api_key_raises_config_error(fake):
    fake([anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=REQ), body=None)])
    with pytest.raises(loop.AgentConfigError, match="API key"):
        loop.run_agent("q", tool_functions={})


def test_transient_api_failure_is_returned_not_raised(fake):
    fake([anthropic.APIConnectionError(request=REQ)])
    r = loop.run_agent("q", tool_functions={})
    assert r["status"] == "api_error" and r["answer"] is None


def test_temperature_only_sent_when_requested(fake):
    client = fake([response([text("a")], "end_turn"), response([text("b")], "end_turn")])
    loop.run_agent("q", tool_functions={})
    loop.run_agent("q", tool_functions={}, temperature=0)
    assert "temperature" not in client.calls[0] and client.calls[1]["temperature"] == 0
