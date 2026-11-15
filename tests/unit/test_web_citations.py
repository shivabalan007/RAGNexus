from types import SimpleNamespace
from uuid import uuid4

from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk


def test_document_and_web_citations_share_a_single_numbering_space():
    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=0, content="Document evidence",
        chunk_metadata={}, document=SimpleNamespace(filename="manual.pdf", document_metadata={}),
    )
    retrieval = RetrievalResult(query="question", chunks=[RetrievedChunk(chunk=chunk)])
    web_sources = [{"title": "Public reference", "url": "https://example.com/source", "content": "Web evidence"}]

    answer, citations = validate_citations("Document fact [1]. External fact [2]. Invalid [4].", retrieval, web_sources)

    assert answer == "Document fact [1]. External fact [2]. Invalid ."
    assert [item["source_type"] for item in citations] == ["document", "web"]
    assert citations[1]["url"] == "https://example.com/source"
