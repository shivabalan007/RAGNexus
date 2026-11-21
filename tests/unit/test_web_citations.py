from types import SimpleNamespace
from uuid import uuid4

from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk


def test_document_and_web_citations_share_a_single_numbering_space():
    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=0,
        content="The document confirms the project uses hybrid retrieval.",
        chunk_metadata={}, document=SimpleNamespace(filename="manual.pdf", document_metadata={}),
    )
    retrieval = RetrievalResult(query="question", chunks=[RetrievedChunk(chunk=chunk)])
    web_sources = [{
        "title": "Public reference", "url": "https://example.com/source",
        "content": "The external web reference confirms the public release date.",
    }]

    answer, citations = validate_citations(
        "The project uses hybrid retrieval [1]. The public release date is confirmed by an external web reference [2]. Invalid [4].",
        retrieval, web_sources,
    )

    assert answer == (
        "The project uses hybrid retrieval [1]. "
        "The public release date is confirmed by an external web reference [2]. "
        "Invalid."
    )
    assert [item["source_type"] for item in citations] == ["document", "web"]
    assert citations[1]["url"] == "https://example.com/source"
