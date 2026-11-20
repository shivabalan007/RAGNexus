from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence


def _normalise(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def retrieval_metrics(
    relevant_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    k: int | None = None,
) -> dict[str, float]:
    """Calculate Precision@K, Recall@K, Hit Rate@K, MRR and nDCG@K."""
    relevant = set(str(x) for x in relevant_ids)
    limit = k if k is not None else len(retrieved_ids)
    if limit is not None and limit < 1:
        raise ValueError("k must be >= 1")
    ranked = [str(x) for x in retrieved_ids[:limit]] if limit is not None else list(retrieved_ids)
    if not relevant:
        return {
            "precision_at_k": 0.0, "recall_at_k": 0.0,
            "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0,
        }
    hits = [1 if item in relevant else 0 for item in ranked]
    precision_denominator = limit if limit is not None else max(len(ranked), 1)
    precision = sum(hits) / max(precision_denominator, 1)
    recall = sum(hits) / len(relevant)
    reciprocal_rank = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = min(len(relevant), precision_denominator)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_rate": float(any(hits)),
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _normalise(prediction), _normalise(reference)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(_normalise(prediction)) == " ".join(_normalise(reference)))


def expected_fact_coverage(answer: str, expected_facts: Sequence[str]) -> float:
    """Fraction of expected facts whose normalized tokens are present in the answer."""
    if not expected_facts:
        return 0.0
    answer_tokens = set(_normalise(answer))
    covered = 0
    for fact in expected_facts:
        fact_tokens = set(_normalise(fact))
        if fact_tokens and fact_tokens.issubset(answer_tokens):
            covered += 1
    return covered / len(expected_facts)


def citation_metrics(
    expected_source_ids: Iterable[str],
    cited_source_ids: Sequence[str],
    valid_source_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    expected = set(str(x) for x in expected_source_ids)
    cited = [str(x) for x in cited_source_ids]
    valid = set(str(x) for x in (valid_source_ids if valid_source_ids is not None else cited))
    correct = set(cited) & expected
    valid_cited = [item for item in cited if item in valid]
    return {
        "citation_precision": len(correct) / max(len(cited), 1),
        "citation_recall": len(correct) / max(len(expected), 1),
        "citation_coverage": float(bool(correct)) if expected else 0.0,
        "source_validity": len(valid_cited) / max(len(cited), 1),
    }


def answerability_metrics(
    *,
    expected_abstain: bool,
    actual_abstain: bool,
    answer_correct: bool | None = None,
) -> dict[str, float]:
    correct_abstention = float(expected_abstain and actual_abstain)
    unsupported_answer = float(expected_abstain and not actual_abstain)
    answerable_accuracy = (
        float((not actual_abstain) and bool(answer_correct))
        if not expected_abstain and answer_correct is not None else 0.0
    )
    return {
        "correct_abstention": correct_abstention,
        "unsupported_answer_rate": unsupported_answer,
        "answerable_question_accuracy": answerable_accuracy,
    }


def context_quality_metrics(
    query: str,
    context_texts: Sequence[str],
    evidence_text: str | None = None,
) -> dict[str, float]:
    """Deterministic lexical proxies; use Ragas judges for semantic variants."""
    contexts = [text for text in context_texts if text]
    query_tokens = set(_normalise(query))
    context_tokens = set(_normalise(" ".join(contexts)))
    evidence_tokens = set(_normalise(evidence_text or ""))
    context_precision = (
        sum(bool(query_tokens & set(_normalise(text))) for text in contexts) / max(len(contexts), 1)
    )
    context_recall = (
        len(evidence_tokens & context_tokens) / max(len(evidence_tokens), 1)
        if evidence_tokens else 0.0
    )
    context_relevancy = len(query_tokens & context_tokens) / max(len(query_tokens), 1)
    noise_sensitivity = (
        sum(not bool(query_tokens & set(_normalise(text))) for text in contexts) / max(len(contexts), 1)
    )
    entity_tokens = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", evidence_text or "")
    }
    context_entity_recall = (
        len(entity_tokens & context_tokens) / max(len(entity_tokens), 1)
        if entity_tokens else 0.0
    )
    return {
        "context_precision_lexical": context_precision,
        "context_recall_lexical": context_recall,
        "context_relevancy_lexical": context_relevancy,
        "noise_sensitivity_lexical": noise_sensitivity,
        "context_entity_recall_lexical": context_entity_recall,
    }


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
