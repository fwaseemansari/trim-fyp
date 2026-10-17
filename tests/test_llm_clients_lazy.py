"""LLMClient: lazy SDK clients, clear missing-key errors, and retry behaviour (no network)."""

import httpx
import groq
import pytest

from pipeline import config, llm_clients
from pipeline.llm_clients import LLMBackend, LLMClient, MissingAPIKeyError


class _Msg:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class FakeCompletions:
    def __init__(self, plan):
        self.plan, self.calls = list(plan), 0

    def create(self, **kwargs):
        self.calls += 1
        step = self.plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return type("R", (), {"choices": [_Msg(step)]})()


def fake_groq_class(completions):
    class FakeGroq:
        def __init__(self, api_key=None):
            self.chat = type("C", (), {"completions": completions})()
    return FakeGroq


def _request():
    return httpx.Request("POST", "http://example.invalid")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_clients.time, "sleep", lambda s: None)


def test_client_builds_without_any_api_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    client = LLMClient()           # used to raise OpenAIError
    assert client.has_openai is False


def test_groq_only_use_never_builds_an_openai_client(monkeypatch):
    comp = FakeCompletions(["ok"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(llm_clients, "Groq", fake_groq_class(comp))
    monkeypatch.setattr(llm_clients, "OpenAI", lambda **kw: pytest.fail("OpenAI client must not be built"))
    assert LLMClient().generate("hi", backend="groq").response == "ok"


@pytest.mark.parametrize("value", [None, "", "your-openai-key-here"])
def test_openai_without_a_key_fails_clearly_when_used(monkeypatch, value):
    monkeypatch.setattr(config, "OPENAI_API_KEY", value)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        LLMClient().generate("hi", backend="openai")


def test_groq_without_a_key_fails_clearly_when_used(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        LLMClient().generate("hi", backend="groq")


def test_connection_errors_are_retried_then_succeed(monkeypatch):
    err = groq.APIConnectionError(request=_request())
    comp = FakeCompletions([err, err, "recovered"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(llm_clients, "Groq", fake_groq_class(comp))
    assert LLMClient().generate("hi", backend="groq").response == "recovered"
    assert comp.calls == 3


def test_rate_limit_is_retried(monkeypatch):
    resp = httpx.Response(429, request=_request())
    limit = groq.RateLimitError("Rate limit. Please try again in 2s.", response=resp, body=None)
    comp = FakeCompletions([limit, "ok"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(llm_clients, "Groq", fake_groq_class(comp))
    assert LLMClient().generate("hi", backend="groq").response == "ok"
    assert comp.calls == 2


def test_other_errors_are_not_retried(monkeypatch):
    comp = FakeCompletions([ValueError("bad request"), "never reached"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(llm_clients, "Groq", fake_groq_class(comp))
    with pytest.raises(ValueError):
        LLMClient().generate("hi", backend="groq")
    assert comp.calls == 1


def test_gives_up_after_max_retries(monkeypatch):
    err = groq.APIConnectionError(request=_request())
    comp = FakeCompletions([err] * 5)
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(llm_clients, "Groq", fake_groq_class(comp))
    backend = LLMClient()._backends[llm_clients.Backend.GROQ]
    with pytest.raises(groq.APIConnectionError):
        backend._call_with_retry("hi", max_retries=3)
    assert comp.calls == 3


def test_retry_wait_parsing():
    assert LLMBackend._parse_retry_wait("try again in 1m47.568s.") == pytest.approx(108.068)
    assert LLMBackend._parse_retry_wait("try again in 350ms") == pytest.approx(0.85)
    assert LLMBackend._parse_retry_wait("something else entirely") is None