"""Orchestration: run steps in order under tracing.

Two entry points:

* :meth:`PipelineRunner.run` - a normal end-to-end execution of one document.
* :meth:`PipelineRunner.run_counterfactual` - replay a finished trace with one step's
  output replaced by a *corrected* one, re-executing everything downstream. This is how
  the analyzer turns "we think step X is the root cause" into a tested claim.

Corpus runs process one document at a time and persist each trace as soon as it
finishes, so memory use stays flat regardless of corpus size - this project is developed
on a machine with about 1 GiB of free RAM.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel

from ffx.config import Settings, get_settings
from ffx.llm.ollama_client import OllamaClient
from ffx.llm.repair import complete_typed
from ffx.llm.replay import ReplayCache
from ffx.models import STEP_ORDER, CorpusDocument, RawDocument, StepName, dump
from ffx.pipeline.steps import STEPS, SYSTEM_PROMPT, build_input, pipeline_version
from ffx.tracing.span import Span, SpanStatus, Trace
from ffx.tracing.store import TraceStore
from ffx.tracing.tracer import StepFailed, Tracer


def build_pipeline_llm(settings: Optional[Settings] = None) -> ReplayCache:
    settings = settings or get_settings()
    return ReplayCache(OllamaClient(settings), enabled=settings.use_cache)


class PipelineRunner:
    def __init__(
        self,
        llm: Any,
        tracer: Tracer,
        settings: Optional[Settings] = None,
    ) -> None:
        self.llm = llm
        self.tracer = tracer
        self.settings = settings or get_settings()
        self.version = pipeline_version()

    # ------------------------------------------------------------ one step

    def _execute(self, step: StepName, step_input: BaseModel) -> BaseModel:
        spec = STEPS[step]
        prompt = spec.build_prompt(step_input)
        with self.tracer.span(step, step_input) as rec:
            outcome = complete_typed(
                self.llm,
                prompt=prompt,
                output_model=spec.output_model,
                schema=spec.schema,
                system=SYSTEM_PROMPT,
                max_tokens=spec.max_tokens,
                max_repair_attempts=self.settings.max_repair_attempts,
            )
            rec.record(
                outcome,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                provider=getattr(self.llm, "provider", ""),
                model=getattr(self.llm, "model", ""),
            )
            if outcome.response is not None:
                rec.span.input_payload.setdefault("_meta", {})
                rec.span.input_payload["_meta"].update(
                    {
                        "prompt_chars": len(prompt),
                        "done_reason": outcome.response.extra.get("done_reason"),
                        "num_ctx": self.settings.num_ctx,
                        "max_tokens": spec.max_tokens,
                    }
                )
        assert outcome.parsed is not None  # tracer.span raises StepFailed otherwise
        return outcome.parsed

    # ------------------------------------------------------------ full run

    def run(
        self,
        doc: RawDocument,
        *,
        run_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Trace:
        with self.tracer.trace(
            doc.doc_id,
            run_id=run_id,
            pipeline_version=self.version,
            metadata={"source_text": doc.source_text, "channel": doc.channel.value, **(metadata or {})},
        ) as trace:
            outputs: dict[StepName, BaseModel] = {}
            for step in STEP_ORDER:
                outputs[step] = self._execute(step, build_input(step, doc, outputs))
        return trace

    def run_corpus(
        self,
        docs: Iterable[CorpusDocument],
        *,
        run_id: str,
        on_trace: Optional[Callable[[Trace, CorpusDocument, float], None]] = None,
    ) -> list[str]:
        trace_ids = []
        for cdoc in docs:
            started = time.perf_counter()
            trace = self.run(
                cdoc.document,
                run_id=run_id,
                metadata={"failure_mode": cdoc.failure_mode.value},
            )
            trace_ids.append(trace.trace_id)
            if on_trace:
                on_trace(trace, cdoc, time.perf_counter() - started)
        return trace_ids

    # ------------------------------------------------------- counterfactual

    def run_counterfactual(
        self,
        original: Trace,
        doc: RawDocument,
        *,
        splice_step: StepName,
        corrected_output: BaseModel,
        reason: str = "",
    ) -> Trace:
        """Re-run ``original`` with ``splice_step``'s output replaced by ``corrected_output``.

        Steps before the splice are copied verbatim (they are not re-executed, so the
        experiment changes exactly one variable). The spliced step gets the corrected
        output. Every step after it is genuinely re-executed by the model.
        """
        splice_index = STEP_ORDER.index(splice_step)
        with self.tracer.trace(
            doc.doc_id,
            run_id=original.run_id,
            pipeline_version=self.version,
            parent_trace_id=original.trace_id,
            metadata={
                **original.metadata,
                "counterfactual": {"splice_step": splice_step.value, "reason": reason},
            },
        ) as trace:
            outputs: dict[StepName, BaseModel] = {}
            for step in STEP_ORDER:
                spec = STEPS[step]
                index = STEP_ORDER.index(step)

                if index < splice_index:
                    source = original.span(step)
                    if source is None or source.output_payload is None:
                        raise StepFailed(step, "cannot splice after a step that produced no output")
                    trace.spans.append(_copy_span(source, trace.trace_id, trace.spans))
                    outputs[step] = spec.output_model.model_validate(source.output_payload)
                    continue

                step_input = build_input(step, doc, outputs)
                if index == splice_index:
                    with self.tracer.span(step, step_input) as rec:
                        rec.set_output(corrected_output)
                        rec.span.spliced = True
                        rec.span.provider = "counterfactual"
                    outputs[step] = corrected_output
                    continue

                outputs[step] = self._execute(step, step_input)
        return trace


def _copy_span(source: Span, trace_id: str, existing: list[Span]) -> Span:
    copy = source.model_copy(deep=True)
    copy.span_id = f"{source.span_id}_cf{len(existing)}_{trace_id[-4:]}"
    copy.trace_id = trace_id
    copy.parent_span_id = existing[-1].span_id if existing else None
    copy.copied_from_span_id = source.span_id
    copy.signals = []
    return copy


def make_runner(store: Optional[TraceStore] = None, exporter: Any = None) -> tuple[PipelineRunner, TraceStore]:
    store = store or TraceStore()
    runner = PipelineRunner(build_pipeline_llm(), Tracer(store, exporter))
    return runner, store


__all__ = ["PipelineRunner", "build_pipeline_llm", "make_runner", "SpanStatus", "dump"]
