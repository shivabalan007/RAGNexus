from collections.abc import Sequence
from .models import Evidence, RetrievalOutput, RetrievalConfig
from .planner import build_plan
from .fusion import reciprocal_rank_fusion
from .sufficiency import assess_sufficiency

class AdvancedRetrievalEngine:
    def __init__(self, retriever, query_llm=None, parent_expander=None,
                 evidence_verifier=None, config: RetrievalConfig | None = None):
        self.retriever = retriever
        self.query_llm = query_llm
        self.parent_expander = parent_expander
        self.evidence_verifier = evidence_verifier
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str, workspace_id: str,
                 document_ids: Sequence[str] | None = None) -> RetrievalOutput:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not workspace_id:
            raise ValueError("workspace_id is required")
        # Tenant and document scoping are passed to every individual search.
        plan = build_plan(query.strip(), self.config, self.query_llm)
        search_queries = list(plan.queries) + list(plan.subquestions)
        search_queries = search_queries[:self.config.max_queries + self.config.max_subquestions]
        rankings = []
        for candidate in search_queries:
            results = self.retriever.search(
                candidate, str(workspace_id),
                [str(x) for x in document_ids] if document_ids is not None else None,
                self.config.top_k_per_query,
            )
            rankings.append(results or [])
        evidence = reciprocal_rank_fusion(rankings, self.config.rrf_k)
        evidence = evidence[:self.config.final_top_k]

        if self.config.enable_parent_expansion and self.parent_expander:
            expanded = []
            for item in evidence:
                try:
                    parent = self.parent_expander.expand_parent(item)
                except Exception:
                    parent = None
                expanded.append(
                    Evidence(
                        chunk_id=item.chunk_id, document_id=item.document_id,
                        content=item.content, rank=item.rank, score=item.score,
                        metadata=item.metadata,
                        parent_content=parent.content if parent else item.parent_content,
                    )
                )
            evidence = expanded

        iterations = 1
        sufficient = assess_sufficiency(evidence, self.config)
        # One bounded second pass: search the original query at a larger top_k.
        # No autonomous recursive LLM loop.
        if (not sufficient and self.config.enable_second_pass
                and self.config.max_iterations > 1):
            extra = self.retriever.search(
                query.strip(), str(workspace_id),
                [str(x) for x in document_ids] if document_ids is not None else None,
                self.config.top_k_per_query * 2,
            ) or []
            evidence = reciprocal_rank_fusion([evidence, extra], self.config.rrf_k)
            evidence = evidence[:self.config.final_top_k]
            iterations += 1
            sufficient = assess_sufficiency(evidence, self.config)

        verification = {}
        if self.evidence_verifier:
            try:
                verification = self.evidence_verifier.verify(query.strip(), evidence) or {}
            except Exception as exc:
                verification = {"status": "verification_unavailable",
                                "reason": type(exc).__name__}
        return RetrievalOutput(
            original_query=query.strip(), plan=plan, evidence=evidence,
            iterations=iterations, sufficient=sufficient,
            verification=verification,
        )
