import json
from pathlib import Path

from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.metrics import citation_metrics, percentile, retrieval_metrics
from ragnexus.evaluation.runner import run_evaluation


def test_100_question_benchmark_is_92_answerable_8_unanswerable():
    path = Path("evals/datasets/ragnexus_ground_truth_100.jsonl")
    cases = load_jsonl(path)
    assert len(cases) == 100
    assert sum(not case.should_abstain for case in cases) == 92
    assert sum(case.should_abstain for case in cases) == 8
    assert all(case.source_document or case.source_documents for case in cases)


def test_unmapped_source_grounded_cases_do_not_fake_retrieval_scores():
    cases = load_jsonl(Path("evals/datasets/ragnexus_ground_truth_100.jsonl"))
    # The user's local benchmark may already have some real chunk mappings.
    # Select an actually-unmapped answerable case rather than assuming row 0
    # will always remain unmapped after the mapping script is run.
    unmapped = next(
        case for case in cases
        if not case.should_abstain and not case.relevant_chunk_ids
    )
    report = run_evaluation(
        [unmapped],
        lambda case: {"answer": case.reference_answer, "abstained": False},
    )
    assert report["retrieval_mapped_case_count"] == 0
    assert report["cases"][0]["retrieval_metrics_status"] == "pending_real_chunk_mapping"
    assert "recall_at_k" not in report["summary"]


def test_citation_metrics_and_percentiles():
    metrics = citation_metrics(["doc-a", "doc-b"], ["doc-a", "doc-x"], ["doc-a", "doc-x"])
    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_recall"] == 0.5
    assert metrics["source_validity"] == 1.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert retrieval_metrics(["a"], ["a"], 1)["precision_at_k"] == 1.0


def test_document_content_hash_unique_index_excludes_deleted_documents():
    import pytest
    pytest.importorskip("pgvector")
    from ragnexus.infrastructure.database.models.document import Document

    index = next(
        item for item in Document.__table__.indexes
        if item.name == "uq_documents_workspace_content_hash_active"
    )
    assert index.unique is True
    assert "deleted" in str(index.dialect_options["postgresql"]["where"])
