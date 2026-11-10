from __future__ import annotations

import re

from ragnexus.retrieval.result import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, retrieval: RetrievalResult) -> tuple[str, list[dict]]:
    """Remove invalid source markers and return only sources cited by the answer."""
    source_count = len(retrieval.chunks)
    used: list[int] = []

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index < 1 or index > source_count:
            return ""
        if index not in used:
            used.append(index)
        return match.group(0)

    cleaned = _CITATION_RE.sub(replace, answer)
    citations: list[dict] = []
    for index in used:
        item = retrieval.chunks[index - 1]
        metadata = item.metadata
        citations.append({
            "source_number": index,
            "document_id": str(item.document_id),
            "chunk_id": str(item.chunk_id),
            "chunk_index": metadata.get("chunk_index"),
            "filename": metadata.get("filename"),
        })
    return cleaned.strip(), citations
