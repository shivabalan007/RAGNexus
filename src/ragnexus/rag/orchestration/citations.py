from __future__ import annotations

import re

from ragnexus.retrieval.result import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, retrieval: RetrievalResult, web_sources: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Validate document and optional web citations in one shared numbering space."""
    web_sources = web_sources or []
    source_count = len(retrieval.chunks) + len(web_sources)
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
    document_count = len(retrieval.chunks)
    for index in used:
        if index <= document_count:
            item = retrieval.chunks[index - 1]
            metadata = item.metadata
            citations.append({
                "source_type": "document", "source_number": index,
                "document_id": str(item.document_id), "chunk_id": str(item.chunk_id),
                "chunk_index": metadata.get("chunk_index"), "filename": metadata.get("filename"),
            })
        else:
            source = web_sources[index - document_count - 1]
            citations.append({
                "source_type": "web", "source_number": index,
                "title": source.get("title") or source.get("url"),
                "url": source.get("url"),
            })
    return cleaned.strip(), citations


def count_active_retrieved_chunks(db, retrieval: RetrievalResult) -> int:
    """Lock source documents and count retrieved chunks that are still active.

    The document row locks are held until the assistant message transaction
    commits. A concurrent soft-delete therefore either happens before this
    validation (and the answer is discarded) or after the message is committed.
    """
    from sqlalchemy import select

    from ragnexus.infrastructure.database.models.document import Document, DocumentChunk

    chunk_ids = [item.chunk_id for item in retrieval.chunks]
    if not chunk_ids:
        return 0
    rows = db.execute(
        select(DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.id.in_(chunk_ids),
            Document.status == "indexed",
            Document.document_metadata["is_current"].as_boolean().is_(True),
        )
        .with_for_update(of=Document)
    ).all()
    return len(rows)
