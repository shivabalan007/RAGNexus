from ragnexus.rag.orchestration.citations import filter_active_citations


def test_history_filter_preserves_web_citations_without_chunk_ids():
    citations = [{
        "source_type": "web", "source_number": 1,
        "title": "Official source", "url": "https://example.com",
    }]
    assert filter_active_citations(citations, set()) == citations


def test_history_filter_keeps_active_supporting_chunk_when_primary_chunk_was_removed():
    document_id = "doc-1"
    deleted_chunk = "chunk-deleted"
    active_chunk = "chunk-active"
    citations = [{
        "source_type": "document", "source_number": 1,
        "document_id": document_id, "chunk_id": deleted_chunk,
        "chunk_index": 4,
        "supporting_chunk_ids": [deleted_chunk, active_chunk],
        "supporting_chunk_indices": [4, 5],
    }]

    filtered = filter_active_citations(
        citations, {(active_chunk, document_id)}
    )

    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == active_chunk
    assert filtered[0]["chunk_index"] == 5
    assert filtered[0]["supporting_chunk_ids"] == [active_chunk]


def test_history_filter_removes_document_citation_when_all_supporting_chunks_are_inactive():
    citations = [{
        "source_type": "document", "source_number": 1,
        "document_id": "doc-1", "chunk_id": "deleted",
    }]
    assert filter_active_citations(citations, set()) == []
