import math

from enum import Enum
from dataclasses import dataclass

from core import retrieval_result


class Route(Enum):
    DOCUMENT = "document"
    WEB = "web"


@dataclass
class RoutingResult:
    route: str
    reason: str
    confidence: float
    confidence_level: str


class QueryRouter:
    """
    Determines whether to answer from uploaded documents
    or trigger web fallback based on retrieval confidence.
    """

    def __init__(
        self,
        high_confidence=0.65,
        medium_confidence=0.40,
        vector_weight=0.6,
        rerank_weight=0.4,
    ):
        self.high_confidence = high_confidence
        self.medium_confidence = medium_confidence

        self.vector_weight = vector_weight
        self.rerank_weight = rerank_weight

    def calculate_confidence(self, retrieval_result) -> float:

        if not retrieval_result.has_documents():
            return 0.0

        chunk_confidences = []

        for item in retrieval_result.retrieved_chunks:

        # Dense retrieval

            vector_confidence = max(0.0, min(item.vector_score, 1.0))

        # BM25

            if item.bm25_score <= 0:
                bm25_confidence = 0.0
            else:
                bm25_confidence = (item.bm25_score /(item.bm25_score + 3.0))

        # CrossEncoder

            rerank_confidence = (1.0 / (1.0 + math.exp(-item.rerank_score)))

        # Combined chunk confidence

            chunk_confidence = (
                0.40 * vector_confidence
                + 0.25 * bm25_confidence
                + 0.35 * rerank_confidence
            )

            chunk_confidences.append(chunk_confidence)

        if not chunk_confidences:
            return 0.0

        confidence = max(chunk_confidences)

        return round(max(0.0, min(confidence, 1.0)),3)
    
    def route(self, retrieval_result) -> RoutingResult:
        """
        Decide whether to answer from the uploaded
        documents or use web fallback.
        """

        if not retrieval_result.has_documents():

            return RoutingResult(
                route=Route.WEB.value,
                reason="No relevant document chunks retrieved.",
                confidence=0.0,
            )

        confidence = self.calculate_confidence(
            retrieval_result
        )

        if confidence >= self.high_confidence:

            return RoutingResult(
                route=Route.DOCUMENT.value,
                reason="High retrieval confidence.",
                confidence=confidence,
                confidence_level="HIGH"
            )

        elif confidence >= self.medium_confidence:

            return RoutingResult(
                route=Route.DOCUMENT.value,
                reason="Medium confidence. Document answer may be incomplete.",
                confidence=confidence,
                confidence_level="MEDIUM"
            )

        else:

            return RoutingResult(
                route=Route.WEB.value,
                reason="Low retrieval confidence. Web fallback recommended.",
                confidence=confidence,
                confidence_level="LOW"
            )


if __name__ == "__main__":

    class DummyRetrievalResult:

        def __init__(self, vector, bm25, rerank):
            self.vector = vector
            self.bm25 = bm25
            self.rerank = rerank

        def has_documents(self):
            return True

        def top_vector_score(self):
            return self.vector

        def top_bm25_score(self):
            return self.bm25

        def top_rerank_score(self):
            return self.rerank

    test_cases = [
        {
            "name": "OOP (Strong Document Match)",
            "vector": 1.62,
            "bm25": 4.20,
            "rerank": 11.59,
        },
        {
            "name": "Machine Learning",
            "vector": 1.54,
            "bm25": 3.47,
            "rerank": 11.19,
        },
        {
            "name": "Docker",
            "vector": 1.42,
            "bm25": 2.90,
            "rerank": 10.69,
        },
        {
            "name": "Weak Retrieval",
            "vector": 0.42,
            "bm25": 0.30,
            "rerank": -2.40,
        },
        {
            "name": "Completely Irrelevant",
            "vector": 0.18,
            "bm25": 0.00,
            "rerank": -8.75,
        },
    ]

    router = QueryRouter()

    print("\n========== ROUTER TEST ==========")

    for test in test_cases:

        retrieval = DummyRetrievalResult(
            vector=test["vector"],
            bm25=test["bm25"],
            rerank=test["rerank"],
        )

        decision = router.route(retrieval)

        print(f"\nTest Case : {test['name']}")
        print(f"Vector Score : {test['vector']}")
        print(f"BM25 Score   : {test['bm25']}")
        print(f"Rerank Score : {test['rerank']}")
        print(f"Route        : {decision.route}")
        print(f"Reason       : {decision.reason}")
        print(f"Confidence   : {decision.confidence:.3f}")
        print(f"Level        : {decision.confidence_level}")
        print("-" * 60)

"""
Confidence-based routing engine for the RAG pipeline. Uses retrieval and
CrossEncoder scores to decide whether to answer from uploaded documents or
trigger web fallback, returning the selected route, confidence, and reason.
"""