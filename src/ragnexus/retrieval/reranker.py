from __future__ import annotations

from threading import RLock
from time import perf_counter
from typing import Protocol

from ragnexus.retrieval.result import RetrievedChunk


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]): ...


# RetrievalService instances can be recreated by dependency factories. Keeping
# the CrossEncoder only on each reranker instance would reload its weights each
# time. Cache the actual scorer process-wide, keyed by model name.
_SCORER_CACHE: dict[str, PairScorer] = {}
_SCORER_CACHE_LOCK = RLock()


def _load_cross_encoder(model_name: str) -> PairScorer:
    """Return one loaded CrossEncoder scorer per model name per process."""
    with _SCORER_CACHE_LOCK:
        scorer = _SCORER_CACHE.get(model_name)
        if scorer is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to enable CrossEncoder reranking"
                ) from exc
            scorer = CrossEncoder(model_name)
            _SCORER_CACHE[model_name] = scorer
        return scorer


class CrossEncoderReranker:
    """Optional CrossEncoder reranker with process-wide lazy model reuse."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: PairScorer | None = None,
    ):
        self.model_name = model_name
        # An explicitly injected scorer remains instance-specific (useful for
        # tests/custom deployments); otherwise _get_scorer uses shared cache.
        self._scorer = scorer

    def _get_scorer(self) -> PairScorer:
        if self._scorer is None:
            self._scorer = _load_cross_encoder(self.model_name)
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
