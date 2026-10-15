"""Corpus size, distribution and reproducibility guarantees."""

from collections import Counter

from ffx.corpus.generate import BENCHMARK_MIX, LEGACY_MIX, generate_corpus


def test_default_corpus_has_expected_220_document_mix():
    docs = generate_corpus(seed=7)

    assert len(docs) == 220
    assert Counter(d.failure_mode for d in docs) == Counter(BENCHMARK_MIX)
    assert len({d.document.doc_id for d in docs}) == 220
    assert docs[0].document.doc_id == "doc_001"
    assert docs[-1].document.doc_id == "doc_220"


def test_original_50_document_prefix_remains_semantically_stable():
    expanded = generate_corpus(seed=7)[:50]
    legacy = generate_corpus(seed=7, mix=LEGACY_MIX)

    for new, old in zip(expanded, legacy):
        assert new.document.doc_id == old.document.doc_id
        assert new.document.source_text == old.document.source_text
        assert new.document.channel == old.document.channel
        assert new.failure_mode == old.failure_mode
        assert new.answer_key == old.answer_key


def test_generation_is_fully_reproducible_for_a_seed():
    first = generate_corpus(seed=19)
    second = generate_corpus(seed=19)

    assert [d.model_dump(mode="json") for d in first] == [d.model_dump(mode="json") for d in second]
