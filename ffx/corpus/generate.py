"""Build the 220-document synthetic benchmark corpus.

Deterministic for a given seed, so the corpus is reproducible and can be committed.
The first 50 documents intentionally preserve the original development corpus: historical
traces refer to those document IDs, so changing their meaning would corrupt benchmark
scoring. The remaining 170 documents broaden every condition for the held-out benchmark.

The mix is weighted towards difficult documents on purpose. ``failure_mode`` describes
the condition injected into the source document, not whether the model actually fails;
clean controls remain numerous enough to measure false positives.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ffx.config import CORPUS_PATH, ensure_dirs
from ffx.corpus.failure_modes import RENDERERS, sample_facts
from ffx.models import CorpusDocument, FailureMode, RawDocument

# The original 50-document development corpus. Keep its counts and generation path
# stable so doc_001 .. doc_050 continue to mean what stored traces say they mean.
LEGACY_MIX: dict[FailureMode, int] = {
    FailureMode.CLEAN: 14,
    FailureMode.DECOY_POLICY_NUMBER: 5,
    FailureMode.CONFLICTING_DATES: 4,
    FailureMode.MISSING_POLICY_NUMBER: 3,
    FailureMode.OCR_GARBLE: 3,
    FailureMode.TRUNCATED: 3,
    FailureMode.OFF_DOMAIN: 3,
    FailureMode.MULTI_CLAIM: 3,
    FailureMode.AMBIGUOUS_PERIL: 3,
    FailureMode.VERBOSE_PADDING: 3,
    FailureMode.NON_ENGLISH_FRAGMENT: 3,
    FailureMode.CONTRADICTORY_AMOUNT: 3,
}

# Final benchmark target: 40 clean controls + 180 difficult documents = 220.
BENCHMARK_MIX: dict[FailureMode, int] = {
    FailureMode.CLEAN: 40,
    FailureMode.DECOY_POLICY_NUMBER: 16,
    FailureMode.CONFLICTING_DATES: 14,
    FailureMode.MISSING_POLICY_NUMBER: 14,
    FailureMode.OCR_GARBLE: 14,
    FailureMode.TRUNCATED: 12,
    FailureMode.OFF_DOMAIN: 12,
    FailureMode.MULTI_CLAIM: 14,
    FailureMode.AMBIGUOUS_PERIL: 14,
    FailureMode.VERBOSE_PADDING: 12,
    FailureMode.NON_ENGLISH_FRAGMENT: 12,
    FailureMode.CONTRADICTORY_AMOUNT: 12,
    FailureMode.SEVERITY_BOUNDARY: 16,
    FailureMode.SPARSE_CLAIM: 18,
}

EXPANSION_MIX: dict[FailureMode, int] = {
    mode: BENCHMARK_MIX[mode] - LEGACY_MIX.get(mode, 0) for mode in BENCHMARK_MIX
}

# Public name retained for callers that imported DEFAULT_MIX.
DEFAULT_MIX = BENCHMARK_MIX

# A fixed clock makes generated JSON byte-for-byte reproducible. The timestamps only
# establish ordering; this corpus is synthetic and does not pretend to be live intake.
CORPUS_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _generate_partition(
    rng: random.Random,
    mix: dict[FailureMode, int],
    *,
    start_index: int,
) -> list[CorpusDocument]:
    plan: list[FailureMode] = [mode for mode, count in mix.items() for _ in range(count)]
    rng.shuffle(plan)

    docs: list[CorpusDocument] = []
    for index, mode in enumerate(plan, start=start_index):
        facts = sample_facts(rng)
        text, channel, answer_key = RENDERERS[mode](facts, rng)
        docs.append(
            CorpusDocument(
                document=RawDocument(
                    doc_id=f"doc_{index:03d}",
                    source_text=text,
                    channel=channel,
                    received_at=CORPUS_EPOCH + timedelta(minutes=index - 1),
                ),
                failure_mode=mode,
                answer_key=answer_key,
            )
        )
    return docs


def generate_corpus(seed: int = 7, mix: Optional[dict[FailureMode, int]] = None) -> list[CorpusDocument]:
    if mix is not None:
        return _generate_partition(random.Random(seed), mix, start_index=1)

    # Preserve the original 50 exactly (apart from making their synthetic timestamps
    # deterministic), then use an independent RNG stream for the expansion. Changing
    # expansion counts later therefore cannot reshuffle or rewrite the legacy prefix.
    legacy = _generate_partition(random.Random(seed), LEGACY_MIX, start_index=1)
    expansion_rng = random.Random(seed ^ 0x5F3759DF)
    expansion = _generate_partition(expansion_rng, EXPANSION_MIX, start_index=len(legacy) + 1)
    return legacy + expansion


def save_corpus(docs: list[CorpusDocument], path: Optional[Path] = None) -> Path:
    ensure_dirs()
    path = path or CORPUS_PATH
    path.write_text(
        json.dumps([d.model_dump(mode="json") for d in docs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_corpus(path: Optional[Path] = None) -> list[CorpusDocument]:
    path = path or CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"No corpus at {path}. Run: python -m ffx corpus generate")
    return [CorpusDocument.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def corpus_index(docs: list[CorpusDocument]) -> dict[str, CorpusDocument]:
    return {d.document.doc_id: d for d in docs}
