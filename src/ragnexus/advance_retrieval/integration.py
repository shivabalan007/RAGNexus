"""Adapter examples for connecting Stage 5 to existing RAGNexus services.

Implement these wrappers against your actual local signatures. Do not modify the
existing retrieval service just to match this example.
"""
from .models import Evidence

class ExistingRetrievalAdapter:
    def __init__(self, retrieve_callable, result_mapper):
        self.retrieve_callable = retrieve_callable
        self.result_mapper = result_mapper

    def search(self, query, workspace_id, document_ids, top_k):
        # retrieve_callable must enforce membership/workspace/document filters in SQL.
        raw = self.retrieve_callable(
            query=query,
            workspace_id=workspace_id,
            document_ids=document_ids,
            top_k=top_k,
        )
        return [self.result_mapper(item, rank=i)
                for i, item in enumerate(raw or [], start=1)]

class LLMQueryAdapter:
    def __init__(self, generate_text):
        self.generate_text = generate_text

    def _parse_lines(self, text, limit):
        values = []
        for line in (text or "").splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip().strip('"')
            if line and line not in values:
                values.append(line)
            if len(values) >= limit:
                break
        return values

    def generate_queries(self, query, max_queries):
        prompt = (
            "Generate alternative search queries for the user question. "
            "Return one query per line, no explanations. Preserve the meaning.\\n"
            f"Question: {query}"
        )
        return self._parse_lines(self.generate_text(prompt), max_queries)

    def decompose(self, query, max_subquestions):
        prompt = (
            "Break this question into the minimum independent answerable subquestions. "
            "Return one subquestion per line, no explanations.\\n"
            f"Question: {query}"
        )
        return self._parse_lines(self.generate_text(prompt), max_subquestions)

class MetadataParentExpander:
    """Example expander when parent text is already available in chunk metadata."""
    def expand_parent(self, evidence):
        parent_text = evidence.metadata.get("parent_content")
        if not parent_text:
            return None
        return Evidence(
            chunk_id=evidence.chunk_id, document_id=evidence.document_id,
            content=parent_text, rank=evidence.rank, score=evidence.score,
            metadata=evidence.metadata,
        )
