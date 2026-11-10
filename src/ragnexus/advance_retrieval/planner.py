import re
from .models import QueryPlan, RetrievalConfig

COMPARISON_TERMS = {
    "compare", "comparison", "versus", " vs ", "difference", "differences",
    "pros and cons", "trade-off", "tradeoff",
}
COMPLEX_TERMS = {
    " and ", "across", "relationship", "impact", "why", "how does",
    "multiple", "both", "all",
}

def analyze_query(query: str) -> str:
    q = f" {re.sub(r'\\s+', ' ', query.lower()).strip()} "
    if any(term in q for term in COMPARISON_TERMS):
        return "comparison"
    if len(q.split()) >= 18 or any(term in q for term in COMPLEX_TERMS):
        return "complex"
    return "simple"

def build_plan(query: str, config: RetrievalConfig, generator=None) -> QueryPlan:
    strategy = analyze_query(query)
    queries = [query]
    subquestions = []
    if generator and config.enable_multi_query and strategy != "simple":
        try:
            generated = generator.generate_queries(query, config.max_queries)
            queries.extend(generated or [])
        except Exception:
            pass
    if generator and config.enable_decomposition and strategy in {"complex", "comparison"}:
        try:
            subquestions = generator.decompose(query, config.max_subquestions) or []
        except Exception:
            subquestions = []
    clean_queries = []
    seen = set()
    for item in queries:
        if isinstance(item, str):
            item = item.strip()
            key = item.casefold()
            if item and key not in seen:
                clean_queries.append(item)
                seen.add(key)
        if len(clean_queries) >= config.max_queries:
            break
    clean_subquestions = []
    seen_sub = set()
    for item in subquestions:
        if isinstance(item, str):
            item = item.strip()
            if item and item.casefold() not in seen_sub:
                clean_subquestions.append(item)
                seen_sub.add(item.casefold())
        if len(clean_subquestions) >= config.max_subquestions:
            break
    return QueryPlan(query, tuple(clean_queries), tuple(clean_subquestions), strategy)
