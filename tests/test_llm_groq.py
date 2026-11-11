"""Groq integration: configuration, request shape, error handling and the number guard (no network)."""
import pytest

from ai import assistant, llm
from core import data_loader as dl
from core.results import build_results


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


@pytest.fixture
def groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    calls = []

    def install(status=200, content="ok"):
        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json})
            return _Resp(status, {"choices": [{"message": {"content": content}}]})
        monkeypatch.setattr(llm.requests, "post", fake_post)
        return calls
    return install


def test_no_key_means_no_llm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.get_config() is None


def test_config_and_request_target_groq(groq_env, monkeypatch):
    calls = groq_env(content="hello")
    cfg = llm.get_config()
    assert cfg.base_url == "https://api.groq.com/openai/v1" and cfg.model == llm.DEFAULT_MODEL
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer gsk_test"
    body = calls[0]["json"]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert body["max_completion_tokens"] == 500 + llm.REASONING_HEADROOM
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    assert llm.get_config().model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("status,needle", [(401, "API key"), (429, "rate limit"), (500, "HTTP 500")])
def test_errors_are_readable(groq_env, status, needle):
    groq_env(status=status)
    with pytest.raises(llm.LLMError, match=needle):
        llm.chat([{"role": "user", "content": "hi"}])


def test_assistant_uses_groq_when_numbers_verify(groq_env):
    df, meta = dl.load_sample("iris")
    r = build_results(df, meta)
    groq_env(content="The dataset has 150 rows and 5 columns.")
    a = assistant.answer("How many rows does the dataset have?", r)
    assert a["mode"] == "llm" and "150" in a["answer"]


def test_assistant_discards_groq_answer_with_invented_numbers(groq_env):
    df, meta = dl.load_sample("iris")
    r = build_results(df, meta)
    groq_env(content="The dataset has 9,999 rows and accuracy is 0.97.")
    a = assistant.answer("How many rows does the dataset have?", r)
    assert a["mode"] == "deterministic" and "9,999" not in a["answer"] and "could not be traced" in a["note"]


def test_rate_limit_falls_back_to_verified_facts(groq_env):
    df, meta = dl.load_sample("iris")
    r = build_results(df, meta)
    groq_env(status=429)
    a = assistant.answer("How many rows does the dataset have?", r)
    assert a["mode"] == "deterministic" and "150" in a["answer"] and "rate limit" in a["note"]
