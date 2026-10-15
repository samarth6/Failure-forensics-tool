"""The judge's Groq -> Gemini -> local chain, tested with fake clients (no network)."""

import json

import pytest

from ffx.analyzer.judge import CLOUD_WEIGHT, LOCAL_WEIGHT, Judge
from ffx.llm.base import LLMError, LLMResponse
from ffx.llm.fallback import FallbackClient
from ffx.llm.replay import ReplayCache


class FakeClient:
    def __init__(self, provider: str, *, fail: bool = False, text: str = '{"choice": "POL-1", "quote": "x"}'):
        self.provider = provider
        self.model = f"{provider}-model"
        self.fail = fail
        self.text = text
        self.calls = 0

    def complete(self, prompt, *, schema=None, system=None, max_tokens=2048):
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.provider} returned 429: rate limited")
        return LLMResponse(text=self.text, model=self.model, provider=self.provider)


def test_first_provider_answers():
    groq, gemini = FakeClient("groq"), FakeClient("gemini")
    r = FallbackClient([groq, gemini]).complete("q")
    assert r.provider == "groq" and r.extra["answered_by"] == "groq"
    assert gemini.calls == 0


def test_rate_limited_provider_falls_through_to_next():
    groq, gemini = FakeClient("groq", fail=True), FakeClient("gemini")
    chain = FallbackClient([groq, gemini])
    r = chain.complete("q")
    assert r.extra["answered_by"] == "gemini"
    assert "429" in r.extra["fell_through"][0]
    assert chain.failures == {"groq": 1, "gemini": 0}


def test_all_providers_failing_raises():
    chain = FallbackClient([FakeClient("groq", fail=True), FakeClient("gemini", fail=True)])
    with pytest.raises(LLMError, match="every judge provider failed"):
        chain.complete("q")


def test_replay_does_not_retry_a_provider_that_failed(tmp_path):
    # Groq was down; Gemini answered. On replay, nothing is called - not even Groq -
    # because the cache is keyed on the chain, not on whoever answered.
    groq, gemini = FakeClient("groq", fail=True), FakeClient("gemini")
    cached = ReplayCache(FallbackClient([groq, gemini]), cache_dir=tmp_path)
    first = cached.complete("q")
    second = cached.complete("q")
    assert first.provider == second.provider == "gemini"
    assert second.from_cache
    assert (groq.calls, gemini.calls) == (1, 1)


@pytest.mark.parametrize("answerer,expected", [("groq", CLOUD_WEIGHT), ("gemini", CLOUD_WEIGHT), ("ollama", LOCAL_WEIGHT)])
def test_judge_weight_follows_whoever_actually_answered(answerer, expected):
    # Cloud answers are independent of the pipeline and earn more trust; a fall-through
    # to the local model is the pipeline grading itself and earns less.
    clients = [FakeClient(p, fail=(p != answerer)) for p in ("groq", "gemini", "ollama")]
    doc = "My policy number is POL-1. I previously held POL-2."
    judge = Judge(FallbackClient(clients), "groq>gemini>local")
    judge.client.clients[-1].text = judge.client.clients[0].text = judge.client.clients[1].text = json.dumps(
        {"choice": "POL-2", "quote": "I previously held POL-2."}
    )
    signal = judge.adjudicate_field("policy_number", ["POL-1", "POL-2"], "POL-1", doc)
    assert signal is not None and not signal.passed
    assert signal.weight == expected
    assert signal.evidence["judge_provider"] == answerer
    assert judge.answered_by == {answerer: 1}
