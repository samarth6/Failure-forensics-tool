"""The LLM judge: last resort, narrow questions only.

The judge is never asked "is this output good?". Holistic 1-5 scores from a small model
are noise. It is only invoked where the deterministic layer has found a *genuine choice*
it cannot settle by string matching - several policy numbers in one document, two dates,
an estimate beside the claimed amount, water vocabulary in an injury claim - and it is
asked a multiple-choice question whose options are exactly those candidates.

Its answers are held to the same standard as the pipeline's: the judge must quote the
sentence that decides the question, and a verdict whose quote is not in the document is
discarded rather than trusted.

Provider: Groq, then Gemini (whichever free-tier keys are configured), then the local
model as the last resort. Every call goes through the replay cache.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ffx.analyzer.grounding import FIELD_KINDS, FieldKind, quote_grounding, values_equal
from ffx.analyzer.signals import make_signal
from ffx.config import Settings, get_settings
from ffx.llm.base import LLMError
from ffx.llm.cloud_judge import PROVIDER_BASE_URLS, OpenAICompatibleClient
from ffx.llm.fallback import FallbackClient
from ffx.llm.ollama_client import OllamaClient
from ffx.llm.replay import ReplayCache
from ffx.models import FailureCategory, Peril, StepName
from ffx.tracing.span import Signal, SignalKind, SignalSeverity, Span

JUDGE_SYSTEM = (
    "You are an auditor checking the work of an automated insurance claims pipeline. "
    "Answer only from the document. Respond with JSON only."
)

FIELD_QUESTIONS = {
    "policy_number": "Which policy number covers THIS loss? Ignore previous, cancelled or unrelated policies.",
    "claim_number": "Which reference number belongs to THIS new claim? Ignore earlier or closed claims.",
    "date_of_loss": "On which date did the loss or damage actually happen? Ignore the date it was reported or written.",
    "reported_amount": "Which amount is the claimant actually claiming? Ignore quotes, estimates, upgrades or earlier payments.",
}


# Generous on purpose. The Groq judge (gpt-oss-120b) is a reasoning model: it spends
# output tokens thinking before it writes JSON. Measured: at 20 tokens it returned an
# empty generation; a trivial one-field answer used 94. A tight budget does not fail
# loudly - it truncates the JSON and silently falls through to the next provider.
JUDGE_MAX_TOKENS = 2048

CLOUD_PROVIDERS = {"groq", "gemini"}
CLOUD_WEIGHT = 0.7
LOCAL_WEIGHT = 0.5


class Judge:
    def __init__(self, client: Any, provider: str) -> None:
        self.client = client
        self.provider = provider  # the roster, for display, e.g. "groq>gemini>local"
        self.calls = 0
        self.answered_by: dict[str, int] = {}  # which provider actually served each call

    # ------------------------------------------------------------------ core

    def _ask(self, prompt: str, schema: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Ask the judge; the returned dict carries ``_answered_by`` (which provider replied)."""
        self.calls += 1
        try:
            response = self.client.complete(prompt, schema=schema, system=JUDGE_SYSTEM, max_tokens=JUDGE_MAX_TOKENS)
        except LLMError:
            return None
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        answered_by = response.extra.get("answered_by", response.provider)
        self.answered_by[answered_by] = self.answered_by.get(answered_by, 0) + 1
        data["_answered_by"] = answered_by
        return data

    def _weight(self, answer: dict[str, Any]) -> float:
        """Trust depends on who actually answered, not on what was configured.

        A cloud model is independent of the pipeline and stronger, so its verdict counts
        for more. If the chain fell through to the local model, that is the pipeline's
        own model grading itself, and it gets the lower weight.
        """
        return CLOUD_WEIGHT if answer.get("_answered_by") in CLOUD_PROVIDERS else LOCAL_WEIGHT

    # ------------------------------------------------------- field candidates

    def adjudicate_field(self, field: str, candidates: list[str], chosen: str, document: str) -> Optional[Signal]:
        options = list(dict.fromkeys(candidates)) + ["none"]
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": options},
                "quote": {"type": "string"},
            },
            "required": ["choice", "quote"],
            "additionalProperties": False,
        }
        prompt = (
            f"QUESTION: {FIELD_QUESTIONS[field]}\n"
            f"OPTIONS: {options}\n"
            "Choose one option, and copy EXACTLY the sentence from the document that decides it.\n\n"
            f"DOCUMENT:\n<<<\n{document}\n>>>\n"
        )
        answer = self._ask(prompt, schema)
        if not answer or answer.get("choice") not in options:
            return None

        choice, quote = answer["choice"], answer.get("quote", "")
        kind = FIELD_KINDS[field]
        weight = self._weight(answer)
        base = {
            "field": field, "candidates": candidates, "chosen": chosen, "judge_choice": choice,
            "judge_quote": quote, "judge_provider": answer["_answered_by"],
        }

        if not quote_grounding(quote, document).grounded:
            return make_signal(
                f"judge:{field}",
                passed=True,
                score=1.0,
                kind=SignalKind.LLM_JUDGE,
                weight=0.0,
                detail=f"judge verdict discarded: its quote is not in the document ({quote[:60]!r})",
                evidence=base,
            )
        if choice == "none":
            return None

        if chosen and values_equal(choice, chosen, kind):
            return make_signal(
                f"judge:{field}",
                passed=True,
                score=1.0,
                kind=SignalKind.LLM_JUDGE,
                weight=weight,
                detail=f"judge agrees: {chosen!r}",
                evidence=base,
            )

        missed = not chosen
        return make_signal(
            f"judge:{field}",
            passed=False,
            score=0.3 if missed else 0.1,
            severity=SignalSeverity.WARN if missed else SignalSeverity.CRITICAL,
            kind=SignalKind.LLM_JUDGE,
            weight=weight,
            detail=(
                f"judge says {field} is {choice!r}, extraction {'left it empty' if missed else f'chose {chosen!r}'}. "
                f"Deciding sentence: {quote[:140]!r}"
            ),
            implicates=FailureCategory.CONTEXT_LOSS if missed else FailureCategory.EXTRACTION_HALLUCINATION,
            evidence=base,
            correction={field: choice},
        )

    # ------------------------------------------------------------------ peril

    def adjudicate_peril(self, chosen: str, document: str) -> Optional[Signal]:
        options = [p.value for p in Peril if p != Peril.UNKNOWN]
        schema = {
            "type": "object",
            "properties": {"peril": {"type": "string", "enum": options}, "quote": {"type": "string"}},
            "required": ["peril", "quote"],
            "additionalProperties": False,
        }
        prompt = (
            "QUESTION: What is the insured peril for this claim? Choose 'liability' when a third "
            "party was injured or suffered loss and is claiming against the insured - even if water "
            "or fire was part of the circumstances. Otherwise choose the direct cause of the damage.\n"
            f"OPTIONS: {options}\n"
            "Copy EXACTLY the sentence from the document that decides it.\n\n"
            f"DOCUMENT:\n<<<\n{document}\n>>>\n"
        )
        answer = self._ask(prompt, schema)
        if not answer or answer.get("peril") not in options:
            return None
        verdict, quote = answer["peril"], answer.get("quote", "")
        weight = self._weight(answer)
        base = {
            "field": "peril", "chosen": chosen, "judge_choice": verdict, "judge_quote": quote,
            "judge_provider": answer["_answered_by"],
        }
        if not quote_grounding(quote, document).grounded:
            return make_signal(
                "judge:peril",
                passed=True,
                score=1.0,
                kind=SignalKind.LLM_JUDGE,
                weight=0.0,
                detail="judge verdict discarded: its quote is not in the document",
                evidence=base,
            )
        if verdict == chosen:
            return make_signal(
                "judge:peril", passed=True, score=1.0, kind=SignalKind.LLM_JUDGE, weight=weight,
                detail=f"judge agrees: {chosen}", evidence=base,
            )
        return make_signal(
            "judge:peril",
            passed=False,
            score=0.1,
            severity=SignalSeverity.CRITICAL,
            kind=SignalKind.LLM_JUDGE,
            weight=weight,
            detail=f"judge says peril is {verdict!r}, extraction chose {chosen!r}. Deciding sentence: {quote[:140]!r}",
            implicates=FailureCategory.MISCLASSIFICATION,
            evidence=base,
            correction={"peril": verdict},
        )

    # ------------------------------------------------------------ dispatcher

    def review(self, span: Span, deterministic: list[Signal]) -> list[Signal]:
        """Ask about every question the deterministic layer flagged on this span."""
        if span.step_name != StepName.EXTRACTION or not span.output_payload:
            return []
        document = span.input_payload.get("normalized_text", "")
        fields = span.output_payload.get("fields", {})
        out: list[Signal] = []
        asked: set[str] = set()
        for sig in deterministic:
            ev = sig.evidence
            if not ev.get("needs_judge"):
                continue
            field = ev.get("field")
            if not field or field in asked:
                continue
            asked.add(field)
            if field == "peril":
                verdict = self.adjudicate_peril(fields.get("peril", "unknown"), document)
            elif field in FIELD_QUESTIONS and len(ev.get("candidates", [])) >= 1:
                verdict = self.adjudicate_field(field, ev["candidates"], fields.get(field, ""), document)
            else:
                verdict = None
            if verdict is not None:
                out.append(verdict)
        return out


def cloud_clients(settings: Settings) -> list[OpenAICompatibleClient]:
    """Configured cloud judges in fallback order: Groq (fastest) first, then Gemini."""
    clients = []
    if settings.groq_api_key:
        clients.append(
            OpenAICompatibleClient(
                base_url=PROVIDER_BASE_URLS["groq"], model=settings.groq_model,
                api_key=settings.groq_api_key, label="groq",
            )
        )
    if settings.gemini_api_key:
        clients.append(
            OpenAICompatibleClient(
                base_url=PROVIDER_BASE_URLS["gemini"], model=settings.gemini_model,
                api_key=settings.gemini_api_key, label="gemini",
            )
        )
    return clients


def build_judge(settings: Optional[Settings] = None) -> Judge:
    """Groq -> Gemini -> local, using whichever cloud keys are configured.

    The local model is always last, so the judge never fails outright just because
    free tiers are rate-limited or down. One replay cache wraps the whole chain - see
    :mod:`ffx.llm.fallback` for why that matters.
    """
    settings = settings or get_settings()
    local = OllamaClient(settings)
    chain: list[Any] = [*cloud_clients(settings), local]
    roster = ">".join(c.provider if c is not local else "local" for c in chain)
    if len(chain) == 1:
        return Judge(ReplayCache(local, enabled=settings.use_cache), roster)
    return Judge(ReplayCache(FallbackClient(chain), enabled=settings.use_cache), roster)


__all__ = ["Judge", "build_judge", "FieldKind"]
