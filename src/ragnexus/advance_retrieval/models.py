from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    document_id: str
    content: str
    rank: int = 0
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_content: str | None = None

@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    queries: tuple[str, ...]
    subquestions: tuple[str, ...]
    strategy: str

@dataclass
class RetrievalOutput:
    original_query: str
    plan: QueryPlan
    evidence: list[Evidence]
    iterations: int
    sufficient: bool
    verification: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievalConfig:
    top_k_per_query: int = 8
    final_top_k: int = 8
    max_queries: int = 4
    max_subquestions: int = 3
    max_iterations: int = 2
    rrf_k: int = 60
    min_evidence: int = 2
    min_distinct_documents: int = 1
    enable_multi_query: bool = True
    enable_decomposition: bool = True
    enable_parent_expansion: bool = False
    enable_second_pass: bool = True

    def __post_init__(self):
        positive = (
            self.top_k_per_query, self.final_top_k, self.max_queries,
            self.max_subquestions, self.max_iterations, self.rrf_k,
            self.min_evidence, self.min_distinct_documents,
        )
        if any(x < 1 for x in positive):
            raise ValueError("Retrieval limits must be positive integers.")
        if self.max_iterations > 3:
            raise ValueError("max_iterations cannot exceed 3.")
