from ragnexus.advanced_retrieval import AdvancedRetrievalEngine, RetrievalConfig
from ragnexus.advanced_retrieval.models import Evidence

class FakeRetriever:
    def __init__(self):
        self.calls = []
    def search(self, query, workspace_id, document_ids, top_k):
        self.calls.append((query, workspace_id, document_ids, top_k))
        return [
            Evidence("c1", "doc1", "evidence one"),
            Evidence("c2", "doc1", "evidence two"),
        ]

def test_engine_passes_tenant_and_document_scope_to_every_search():
    retriever = FakeRetriever()
    engine = AdvancedRetrievalEngine(
        retriever,
        config=RetrievalConfig(enable_multi_query=False,
                               enable_decomposition=False,
                               enable_second_pass=False),
    )
    result = engine.retrieve("Explain retrieval", "workspace-1", ["doc1"])
    assert result.sufficient is True
    assert retriever.calls
    assert all(call[1] == "workspace-1" for call in retriever.calls)
    assert all(call[2] == ["doc1"] for call in retriever.calls)

def test_empty_query_rejected():
    engine = AdvancedRetrievalEngine(FakeRetriever())
    try:
        engine.retrieve(" ", "workspace-1")
        assert False, "Expected ValueError"
    except ValueError:
        pass
