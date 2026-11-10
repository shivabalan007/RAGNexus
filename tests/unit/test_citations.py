from types import SimpleNamespace
from uuid import uuid4

from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk


def test_validate_citations_removes_unknown_sources():
    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=4, content="Evidence",
        chunk_metadata={}, document=SimpleNamespace(filename="guide.pdf", document_metadata={}),
    )
    retrieval = RetrievalResult(query="question", chunks=[RetrievedChunk(chunk=chunk)])
    answer, citations = validate_citations("Fact [1]. Bad reference [8].", retrieval)
    assert answer == "Fact [1]. Bad reference ."
    assert len(citations) == 1
    assert citations[0]["filename"] == "guide.pdf"
