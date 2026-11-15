from __future__ import annotations

import re

from ragnexus.retrieval.result import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_citations(
    answer: str,
    retrieval: RetrievalResult,
    web_sources: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Keep only citations that map to concrete document or Tavily evidence."""
    docs = list(retrieval.chunks)
    web = list(web_sources or [])
    source_count = len(docs) + len(web)
    used: list[int] = []

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index < 1 or index > source_count:
            return ""
        if index not in used:
            used.append(index)
        return match.group(0)

    cleaned = _CITATION_RE.sub(replace, answer or "")
    citations: list[dict] = []
    for index in used:
        if index <= len(docs):
            item = docs[index - 1]
            metadata = item.metadata
            citations.append({
                "source_number": index,
                "source_type": "document",
                "document_id": str(item.document_id),
                "chunk_id": str(item.chunk_id),
                "chunk_index": metadata.get("chunk_index"),
                "filename": metadata.get("filename"),
                "url": None,
                "title": metadata.get("filename"),
                "excerpt": item.content[:1200],
            })
        else:
            source = web[index - len(docs) - 1]
            citations.append({
                "source_number": index,
                "source_type": "web",
                "document_id": None,
                "chunk_id": None,
                "chunk_index": None,
                "filename": source.get("title"),
                "url": source.get("url"),
                "title": source.get("title"),
                "excerpt": str(source.get("content", ""))[:1200],
            })
    return cleaned.strip(), citations
