from types import SimpleNamespace
from uuid import uuid4

from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk


def test_validate_citations_removes_unknown_sources():
    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=4,
        content="Python is an interpreted programming language.",
        chunk_metadata={}, document=SimpleNamespace(filename="guide.pdf", document_metadata={}),
    )
    retrieval = RetrievalResult(query="question", chunks=[RetrievedChunk(chunk=chunk)])
    answer, citations = validate_citations(
        "Python is an interpreted programming language [1]. Bad reference [8].", retrieval
    )
    assert answer == "Python is an interpreted programming language [1]. Bad reference."
    assert len(citations) == 1
    assert citations[0]["filename"] == "guide.pdf"


def test_wrong_chunk_citation_is_rejected_and_same_document_is_deduplicated():
    document_id = uuid4()
    weak_chunk = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=4,
        content="NovaTech was founded in 2018.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    supporting_chunk = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=0,
        content="NovaTech Solutions was founded in 2018 by Elena Rodriguez and Marcus Chen in Portland, Oregon.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(
        query="Who founded NovaTech?",
        chunks=[RetrievedChunk(chunk=weak_chunk), RetrievedChunk(chunk=supporting_chunk)],
    )

    answer, citations = validate_citations(
        "NovaTech Solutions was founded in 2018 by Elena Rodriguez and Marcus Chen in Portland, Oregon [1][2].",
        retrieval,
    )

    assert answer.endswith("[1].")
    assert len(citations) == 1
    assert citations[0]["document_id"] == str(document_id)
    assert citations[0]["chunk_id"] == str(supporting_chunk.id)
    assert citations[0]["supporting_chunk_ids"] == [str(supporting_chunk.id)]



def test_multiple_chunks_can_jointly_support_one_sentence_without_duplicate_document_citations():
    document_id = uuid4()
    first = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=0,
        content="NovaTech was founded in 2018.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    second = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=1,
        content="NovaTech currently employs 247 people.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(
        query="How old is NovaTech and how many employees?",
        chunks=[RetrievedChunk(chunk=first), RetrievedChunk(chunk=second)],
    )

    answer, citations = validate_citations(
        "NovaTech was founded in 2018 and employs 247 people [1][2].",
        retrieval,
    )

    assert answer == "NovaTech was founded in 2018 and employs 247 people [1]."
    assert len(citations) == 1
    assert set(citations[0]["supporting_chunk_ids"]) == {str(first.id), str(second.id)}



def test_attribution_ignores_generator_citation_and_selects_only_supporting_chunk():
    from ragnexus.rag.orchestration.citations import attribute_citations
    from ragnexus.llm.providers.base import LLMResponse

    doc_id = uuid4()
    unrelated = SimpleNamespace(
        id=uuid4(), document_id=doc_id, chunk_index=4,
        content="NovaTech has offices in Portland.", chunk_metadata={},
        document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    supporting = SimpleNamespace(
        id=uuid4(), document_id=doc_id, chunk_index=0,
        content="NovaTech Solutions was founded in 2018 by Elena Rodriguez and Marcus Chen in Portland, Oregon.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(query="Who founded NovaTech?", chunks=[
        RetrievedChunk(chunk=unrelated), RetrievedChunk(chunk=supporting),
    ])

    class Auditor:
        def complete(self, *_args, **_kwargs):
            return LLMResponse('{"items":[{"segment":1,"source_numbers":[2]}]}', "test-model")

    answer, citations, verified = attribute_citations(
        "NovaTech Solutions was founded by Elena Rodriguez and Marcus Chen [1][2].",
        retrieval, [], Auditor(),
    )
    assert verified is True
    assert answer.endswith("[1].")
    assert len(citations) == 1
    assert citations[0]["document_id"] == str(doc_id)
    assert citations[0]["supporting_chunk_ids"] == [str(supporting.id)]


def test_attribution_deduplicates_same_document_but_retains_supporting_chunks():
    from ragnexus.rag.orchestration.citations import attribute_citations
    from ragnexus.llm.providers.base import LLMResponse

    doc_id = uuid4()
    chunks = []
    for index, content in enumerate(["NovaTech was founded in 2018.", "NovaTech was founded by Elena Rodriguez."]):
        chunk = SimpleNamespace(
            id=uuid4(), document_id=doc_id, chunk_index=index, content=content,
            chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
        )
        chunks.append(RetrievedChunk(chunk=chunk))
    retrieval = RetrievalResult(query="Who founded it?", chunks=chunks)

    class Auditor:
        def complete(self, *_args, **_kwargs):
            return LLMResponse('{"items":[{"segment":1,"source_numbers":[1,2]}]}', "test-model")

    answer, citations, verified = attribute_citations(
        "NovaTech was founded in 2018 by Elena Rodriguez.", retrieval, [], Auditor()
    )
    assert verified is True
    assert answer.endswith("[1].")
    assert len(citations) == 1
    assert len(citations[0]["supporting_chunk_ids"]) == 2


def test_attribution_preserves_answer_when_no_source_supports_claim():
    from ragnexus.rag.orchestration.citations import attribute_citations
    from ragnexus.llm.providers.base import LLMResponse

    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=0,
        content="The company has 247 employees.", chunk_metadata={},
        document=SimpleNamespace(filename="company.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(query="Who is the CEO?", chunks=[RetrievedChunk(chunk=chunk)])

    class Auditor:
        def complete(self, *_args, **_kwargs):
            return LLMResponse('{"items":[{"segment":1,"source_numbers":[]}]}', "test-model")

    answer, citations, verified = attribute_citations("The CEO is Alex Smith.", retrieval, [], Auditor())
    assert answer == "The CEO is Alex Smith."
    assert citations == []
    assert verified is False


def test_citation_mapping_finds_exact_supporting_chunk_even_if_generator_cites_wrong_chunk():
    document_id = uuid4()
    weak = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=4,
        content="NovaTech was founded in 2018.", chunk_metadata={},
        document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    exact = SimpleNamespace(
        id=uuid4(), document_id=document_id, chunk_index=0,
        content="NovaTech Solutions was founded in 2018 by Elena Rodriguez and Marcus Chen in Portland, Oregon.",
        chunk_metadata={}, document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(query="Who founded NovaTech?", chunks=[
        RetrievedChunk(chunk=weak), RetrievedChunk(chunk=exact),
    ])

    answer, citations = validate_citations(
        "NovaTech Solutions was founded in 2018 by Elena Rodriguez and Marcus Chen [1].",
        retrieval,
    )

    assert answer.endswith("[1].")
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == str(exact.id)
    assert citations[0]["supporting_chunk_ids"] == [str(exact.id)]


def test_citation_mapping_never_calls_a_second_model():
    from ragnexus.rag.orchestration.citations import attribute_citations

    chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=0,
        content="NovaTech currently employs 247 people.", chunk_metadata={},
        document=SimpleNamespace(filename="novatech.docx", document_metadata={}),
    )
    retrieval = RetrievalResult(query="How many employees?", chunks=[RetrievedChunk(chunk=chunk)])

    class ModelThatMustNotBeCalled:
        def complete(self, *_args, **_kwargs):
            raise AssertionError("Citation mapping must not call the chat model")

    answer, citations, attributed = attribute_citations(
        "NovaTech has 247 employees [99].", retrieval, [], ModelThatMustNotBeCalled()
    )

    assert answer.endswith("[1].")
    assert len(citations) == 1
    assert attributed is True
