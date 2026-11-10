from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence


def _normalise(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def retrieval_metrics(relevant_ids: Iterable[str], retrieved_ids: Sequence[str], k: int | None = None) -> dict[str, float]:
    """Calculate standard ranking metrics using document/chunk IDs as judged units."""
    relevant = set(relevant_ids)
    ranked = list(retrieved_ids[:k] if k is not None else retrieved_ids)
    if not relevant:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    hits = [1 if item in relevant else 0 for item in ranked]
    precision = sum(hits) / max(len(ranked), 1)
    recall = sum(hits) / len(relevant)
    reciprocal_rank = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal_hits = min(len(relevant), len(ranked))
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
    from collections import Counter
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(_normalise(prediction)) == " ".join(_normalise(reference)))
