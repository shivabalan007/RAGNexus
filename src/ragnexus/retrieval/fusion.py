from __future__ import annotations

from collections import defaultdict
from typing import Hashable

from ragnexus.retrieval.result import RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[tuple[object, float]],
    sparse_results: list[tuple[object, float]],
    *,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked lists using RRF; raw dense/BM25 scores are not comparable."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    by_id: dict[Hashable, RetrievedChunk] = {}
    scores: dict[Hashable, float] = defaultdict(float)

    for source, rows in (("dense", dense_results), ("sparse", sparse_results)):
        for rank, (chunk, raw_score) in enumerate(rows, start=1):
            key = chunk.id
            if key not in by_id:
                by_id[key] = RetrievedChunk(chunk=chunk)
            item = by_id[key]
            scores[key] += 1.0 / (rrf_k + rank)
            if source == "dense":
                item.dense_score = float(raw_score)
            else:
                item.bm25_score = float(raw_score)

    for key, item in by_id.items():
        item.rrf_score = scores[key]

    return sorted(by_id.values(), key=lambda item: item.rrf_score, reverse=True)
