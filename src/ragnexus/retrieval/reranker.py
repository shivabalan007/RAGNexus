from __future__ import annotations

from time import perf_counter
from typing import Protocol

from ragnexus.retrieval.result import RetrievedChunk


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]): ...


class CrossEncoderReranker:
    """Optional CrossEncoder reranker. Model is loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: PairScorer | None = None,
    ):
        self.model_name = model_name
        self._scorer = scorer

    def _get_scorer(self) -> PairScorer:
        if self._scorer is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to enable CrossEncoder reranking"
                ) from exc
            self._scorer = CrossEncoder(self.model_name)
        return self._scorer

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        started = perf_counter()
        if not candidates:
            return [], (perf_counter() - started) * 1000
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scorer = self._get_scorer()
        pairs = [(query, item.content) for item in candidates]
        scores = scorer.predict(pairs)
        if len(scores) != len(candidates):
            raise RuntimeError("CrossEncoder returned a mismatched score count")
        for item, score in zip(candidates, scores):
            item.rerank_score = float(score)
        candidates.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        return candidates[:top_k], (perf_counter() - started) * 1000



"""
Uses CrossEncoder (ms-marco-MiniLM-L-6-v2) to rerank retrieved chunks and
updates RetrievalResult with rerank scores while preserving chunk-to-score
relationships for routing, monitoring, and memory.
"""