from types import SimpleNamespace
from uuid import uuid4

from ragnexus.ingestion.base import LoadedDocument
from ragnexus.ingestion.chunking.structure_aware import StructureAwareChunker
from ragnexus.retrieval.advanced import (
    ContextualCompressor,
    EvidenceSufficiencyChecker,
    MultiQueryGenerator,
)
from ragnexus.retrieval.tokenization import tokenize


def test_structure_aware_chunker_preserves_long_text_and_parent_metadata():
    long_sentence = " ".join(f"token{i}" for i in range(900))
    source = f"# Temperature Monitoring\n\n{long_sentence}\n\n## Escalation\n\nContact Operations Team."
    chunks = StructureAwareChunker(chunk_size=240, parent_max_chars=900).split(
        [LoadedDocument(source, {"source_filename": "guide.md", "page_number": 1})]
    )

    assert len(chunks) > 3
    chunk_tokens = [token for chunk in chunks for token in tokenize(chunk.text)]
    source_tokens = tokenize(long_sentence + " Contact Operations Team.")
    assert set(source_tokens).issubset(set(chunk_tokens))
    assert all(chunk.metadata["chunk_type"] == "child" for chunk in chunks)
    assert all(chunk.metadata["parent_id"] for chunk in chunks)
    assert all(chunk.metadata["parent_content"] for chunk in chunks)
    assert any(chunk.metadata["section_title"] == "Escalation" for chunk in chunks)


def test_multi_query_generator_deduplicates_and_obeys_maximum():
    generator = MultiQueryGenerator(
        generate=lambda _: "alternate one\nalternate one\n- alternate two\nextra",
        max_queries=3,
    )
    assert generator.generate_queries("original question") == [
        "original question", "alternate one", "alternate two"
    ]


def test_multi_query_generator_falls_back_to_original_on_provider_error():
    def fail(_):
        raise RuntimeError("provider offline")

    assert MultiQueryGenerator(generate=fail).generate_queries("keep this query") == ["keep this query"]


def test_contextual_compression_keeps_relevant_evidence():
    source = (
        "The warehouse uses Sensor T-17. "
        "The office kitchen closes at 6 PM. "
        "The temperature alert threshold is 8 degrees Celsius. "
        "The parking gate is blue. "
    )
    compressed = ContextualCompressor(max_sentences=2, max_chars=300).compress(
        "warehouse temperature alert threshold", source
    )
    assert "Sensor T-17" in compressed
    assert "8 degrees Celsius" in compressed
    assert "office kitchen" not in compressed


def test_evidence_sufficiency_checker_distinguishes_missing_evidence():
    checker = EvidenceSufficiencyChecker(min_token_overlap=0.2)
    relevant = SimpleNamespace(content="Falcon monitors warehouse temperature using Sensor T-17.", dense_score=None, rerank_score=None)
    irrelevant = SimpleNamespace(content="The office cafeteria menu changes every Friday.", dense_score=None, rerank_score=None)

    assert checker.assess("What sensor monitors warehouse temperature?", [relevant])[0] is True
    sufficient, reason = checker.assess("What is the warehouse temperature threshold?", [irrelevant])
    assert sufficient is False
    assert reason.startswith("weak_evidence")


def test_evidence_sufficiency_respects_disabled_flag():
    checker = EvidenceSufficiencyChecker()
    item = SimpleNamespace(content="", dense_score=None, rerank_score=None)
    assert checker.assess("anything", [item], enabled=False) == (True, "check_disabled")


def test_contextual_compressor_does_not_truncate_single_oversized_evidence_sentence():
    source = "Sensor T-17 threshold " + ("critical " * 100)
    compressed = ContextualCompressor(max_sentences=2, max_chars=200).compress(
        "Sensor T-17 threshold", source
    )
    assert compressed == source.strip()
