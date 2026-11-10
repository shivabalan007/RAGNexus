from ragnexus.evaluation.metrics import exact_match, retrieval_metrics, token_f1


def test_retrieval_metrics_perfect_ranking():
    result = retrieval_metrics(["a", "b"], ["a", "b"], 2)
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg"] == 1.0


def test_retrieval_metrics_miss():
    result = retrieval_metrics(["a"], ["x", "y"], 2)
    assert result["hit_rate"] == 0.0
    assert result["recall_at_k"] == 0.0


def test_answer_metrics():
    assert exact_match("Hello, WORLD!", "hello world") == 1.0
    assert token_f1("a b c", "a b") > 0.0
    assert token_f1("alpha", "beta") == 0.0
