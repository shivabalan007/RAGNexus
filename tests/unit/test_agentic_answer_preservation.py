from types import SimpleNamespace
from uuid import uuid4

from ragnexus.agentic.graph import AgenticRAGWorkflow
from ragnexus.llm.providers.base import LLMResponse
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk


def _retrieval():
    document_id = uuid4()
    chunk = SimpleNamespace(
        id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        content="NovaTech currently employs 247 people across three offices.",
        chunk_metadata={},
        document=SimpleNamespace(
            filename="NovaTech-Solution.docx",
            document_metadata={},
        ),
    )
    return RetrievalResult(
        query="How many employees does NovaTech have?",
        chunks=[RetrievedChunk(chunk=chunk)],
        evidence_sufficient=False,
        evidence_reason="grader_false_negative",
    )


def test_agentic_generation_runs_when_chunks_exist_even_if_grader_says_insufficient():
    class ChatModel:
        def complete(self, *_args, **_kwargs):
            return LLMResponse(
                "NovaTech currently employs 247 people [1].",
                "test-model",
                prompt_tokens=10,
                completion_tokens=8,
            )

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.chat_model = ChatModel()
    retrieval = _retrieval()
    result = workflow._generate({
        "original_query": "How many employees does NovaTech have?",
        "query": "How many employees does NovaTech have?",
        "route": "documents",
        "retrieval": retrieval,
        "retrieval_sufficient": False,
        "web_sources": [],
        "history": [],
    })

    assert result["generation_attempted"] is True
    assert "247" in result["answer"]
    assert result["abstained"] is False
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == str(retrieval.chunks[0].chunk_id)


def test_agentic_verifier_false_negative_does_not_erase_answer():
    class FalseNegativeVerifier:
        def verify(self, **_kwargs):
            return {"status": "completed", "verified": False, "reason": "model_false_negative"}

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.settings = SimpleNamespace(AGENTIC_ENABLE_ANSWER_VERIFICATION=True)
    workflow.verifier = FalseNegativeVerifier()
    retrieval = _retrieval()
    state = {
        "original_query": "How many employees does NovaTech have?",
        "route": "documents",
        "retrieval": retrieval,
        "answer": "NovaTech currently employs 247 people.",
        "citations": [{"source_number": 1, "chunk_id": str(retrieval.chunks[0].chunk_id)}],
    }

    update = workflow._verify(state)

    assert "answer" not in update
    assert "citations" not in update
    assert update["verification"]["verified"] is False
