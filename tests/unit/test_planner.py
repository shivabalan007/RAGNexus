from ragnexus.advanced_retrieval.models import RetrievalConfig
from ragnexus.advanced_retrieval.planner import analyze_query, build_plan

class FakeGenerator:
    def generate_queries(self, query, max_queries):
        return ["alternate query", "alternate query"]
    def decompose(self, query, max_subquestions):
        return ["part one", "part two", "part three", "part four"]

def test_simple_query_uses_simple_strategy():
    assert analyze_query("What is pgvector?") == "simple"

def test_comparison_query_generates_bounded_plan():
    config = RetrievalConfig(max_queries=3, max_subquestions=2)
    plan = build_plan("Compare dense and sparse retrieval approaches", config, FakeGenerator())
    assert plan.strategy == "comparison"
    assert len(plan.queries) <= 3
    assert len(plan.subquestions) <= 2
    assert len(plan.queries) == len(set(plan.queries))
