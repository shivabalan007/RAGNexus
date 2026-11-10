from ragnexus.retrieval.tokenization import tokenize


def test_tokenizer_is_case_insensitive_and_empty_safe():
    assert tokenize("RAG Systems are useful") == ["rag", "systems", "are", "useful"]
    assert tokenize("") == []
