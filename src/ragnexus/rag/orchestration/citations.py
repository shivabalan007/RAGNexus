from __future__ import annotations

import re
from collections import OrderedDict
from urllib.parse import urlsplit, urlunsplit

from ragnexus.retrieval.result import RetrievalResult

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_PROPER_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9&'.-]*)(?:\s+[A-Z][a-zA-Z0-9&'.-]*)*\b"
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
    "company", "according", "document", "documents",
}
_SENTENCE_START_GENERIC = {
    "a", "an", "the", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "i", "according", "based",
}


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(value or "")
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def _normalise_source_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        parsed = urlsplit(url.strip())
        return urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        ))
    except Exception:
        return url.strip()


def _claim_requirements(claim: str) -> tuple[set[str], set[str]]:
    """Return exact numeric and named-entity tokens that a source must contain."""
    numeric = {
        value.casefold().replace(",", "")
        for value in _NUMBER_RE.findall(claim or "")
    }
    entity_tokens: set[str] = set()
    for phrase in _PROPER_PHRASE_RE.findall(claim or ""):
        raw_words = _TOKEN_RE.findall(phrase)
        words = _tokens(phrase)
        if not words:
            continue
        first = raw_words[0].casefold() if raw_words else ""
        if first in _SENTENCE_START_GENERIC:
            words.discard(first)
        # All-uppercase labels such as CEO are not named entities.
        uppercase_labels = {word.casefold() for word in raw_words if word.isupper()}
        words.difference_update(uppercase_labels)
        entity_tokens.update(words)
    return numeric, entity_tokens


def _source_records(
    retrieval: RetrievalResult,
    web_sources: list[dict] | None,
) -> list[dict]:
    records: list[dict] = []
    for item in retrieval.chunks:
        metadata = item.metadata or {}
        records.append({
            "source_type": "document",
            "document_id": str(item.document_id),
            "chunk_id": str(item.chunk_id),
            "chunk_index": metadata.get("chunk_index"),
            "filename": metadata.get("filename"),
            "text": item.context_content or "",
            "tokens": _tokens(item.context_content or ""),
            "numbers": {
                value.casefold().replace(",", "")
                for value in _NUMBER_RE.findall(item.context_content or "")
            },
        })
    for source in web_sources or []:
        evidence = f"{source.get('title', '')} {source.get('content', '')}"
        records.append({
            "source_type": "web",
            "title": source.get("title") or source.get("url") or "Web source",
            "url": source.get("url"),
            "text": evidence,
            "tokens": _tokens(evidence),
            "numbers": {
                value.casefold().replace(",", "")
                for value in _NUMBER_RE.findall(evidence)
            },
        })
    return records


def _select_supporting_sources(claim: str, sources: list[dict]) -> list[int]:
    """Select supporting chunks from one source identity, never combine unrelated docs.

    This is deliberately deterministic: it never calls the answer LLM a second
    time. Numeric values and named entities must occur in the selected evidence.
    Token overlap is a relevance filter, not a claim of semantic entailment.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return []

    required_numbers, required_entities = _claim_requirements(claim)
    required_terms = required_numbers | required_entities
    minimum_coverage = 0.25 if len(claim_tokens) < 5 else 0.20
    target_coverage = max(1, int(len(claim_tokens) * minimum_coverage + 0.999))

    # Chunks may be combined only when they belong to the same document.
    # Web results are grouped by canonical URL. This avoids joining an entity
    # mention from one file with a number from an unrelated file.
    groups: OrderedDict[tuple[str, str], list[tuple[int, dict]]] = OrderedDict()
    for index, source in enumerate(sources):
        if source["source_type"] == "document":
            identity = ("document", source["document_id"])
        else:
            identity = (
                "web",
                _normalise_source_url(source.get("url")) or source.get("title", ""),
            )
        groups.setdefault(identity, []).append((index, source))

    viable: list[tuple[float, list[int]]] = []
    for members in groups.values():
        overlaps = [
            (index, source["tokens"] & claim_tokens)
            for index, source in members
        ]
        overlaps = [(index, overlap) for index, overlap in overlaps if overlap]
        if not overlaps:
            continue

        selected: list[int] = []
        covered: set[str] = set()
        remaining = list(overlaps)
        while remaining:
            best_position = max(
                range(len(remaining)),
                key=lambda pos: (
                    len(remaining[pos][1] - covered),
                    len(remaining[pos][1]),
                ),
            )
            source_index, source_overlap = remaining.pop(best_position)
            new_terms = source_overlap - covered
            if not new_terms:
                break
            selected.append(source_index)
            covered.update(source_overlap)
            if required_terms.issubset(covered) and len(covered) >= target_coverage:
                break

        if not selected or not required_terms.issubset(covered):
            continue
        coverage = len(covered) / max(1, len(claim_tokens))
        if coverage >= minimum_coverage:
            viable.append((coverage, selected))

    if not viable:
        return []
    # One best source identity per claim keeps citations compact and avoids
    # attaching every retrieved document that happens to share a few words.
    return max(viable, key=lambda item: item[0])[1]


def validate_citations(
    answer: str,
    retrieval: RetrievalResult,
    web_sources: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Rebuild citation provenance from retrieved evidence without another LLM call.

    Generator-provided citation numbers are treated as untrusted. Every answer
    segment is matched against the exact evidence supplied to the generation
    prompt. A failed citation match removes the citation marker, not the answer.
    """
    sources = _source_records(retrieval, web_sources)
    grouped: OrderedDict[tuple[str, str], dict] = OrderedDict()
    rendered: list[str] = []

    for segment in _SENTENCE_RE.split(answer or ""):
        clean_segment = _CITATION_RE.sub("", segment).strip()
        clean_segment = re.sub(r"\s+([.!?,;:])", r"\1", clean_segment)
        if not clean_segment:
            continue

        selected_indices = _select_supporting_sources(clean_segment, sources)
        citation_numbers: list[int] = []

        for source_index in selected_indices:
            source = sources[source_index]
            if source["source_type"] == "document":
                identity = ("document", source["document_id"])
                if identity not in grouped:
                    grouped[identity] = {
                        "source_type": "document",
                        "source_number": len(grouped) + 1,
                        "document_id": source["document_id"],
                        "filename": source.get("filename"),
                        "chunk_id": source["chunk_id"],
                        "chunk_index": source.get("chunk_index"),
                        "supporting_chunk_ids": [],
                        "supporting_chunk_indices": [],
                    }
                citation = grouped[identity]
                if source["chunk_id"] not in citation["supporting_chunk_ids"]:
                    citation["supporting_chunk_ids"].append(source["chunk_id"])
                if source.get("chunk_index") not in citation["supporting_chunk_indices"]:
                    citation["supporting_chunk_indices"].append(source.get("chunk_index"))
            else:
                identity = (
                    "web",
                    _normalise_source_url(source.get("url")) or source.get("title", ""),
                )
                if identity not in grouped:
                    grouped[identity] = {
                        "source_type": "web",
                        "source_number": len(grouped) + 1,
                        "title": source.get("title"),
                        "url": source.get("url"),
                    }
                citation = grouped[identity]

            number = citation["source_number"]
            if number not in citation_numbers:
                citation_numbers.append(number)

        suffix = " ".join(f"[{number}]" for number in citation_numbers)
        if suffix:
            # Put markers before sentence punctuation: "answer [1]."
            match = re.search(r"([.!?]+[\"'’”)]*)$", clean_segment)
            if match:
                punctuation = match.group(1)
                body = clean_segment[:match.start()].rstrip()
                rendered.append(f"{body} {suffix}{punctuation}")
            else:
                rendered.append(f"{clean_segment} {suffix}")
        else:
            rendered.append(clean_segment)

    # Never discard a generated answer merely because citation attribution did
    # not meet the conservative matching threshold.
    return " ".join(rendered), list(grouped.values())


def attribute_citations(
    answer: str,
    retrieval: RetrievalResult,
    web_sources: list[dict] | None = None,
    chat_model=None,
) -> tuple[str, list[dict], bool]:
    """Backward-compatible wrapper; citation attribution is now deterministic.

    ``chat_model`` is intentionally unused. Keeping this wrapper avoids
    breaking external callers while removing the fragile second LLM call.
    """
    cleaned, citations = validate_citations(answer, retrieval, web_sources)
    return cleaned, citations, bool(citations)

def filter_active_citations(
    citations: list[dict],
    valid_pairs: set[tuple[str, str]],
) -> list[dict]:
    """Keep web citations and document citations with active supporting chunks."""
    retained: list[dict] = []
    for original in citations:
        citation = dict(original)
        if citation.get("source_type", "document") == "web":
            retained.append(citation)
            continue

        supporting_ids = citation.get("supporting_chunk_ids") or []
        supporting_indices = citation.get("supporting_chunk_indices") or []
        if not supporting_ids and citation.get("chunk_id"):
            supporting_ids = [citation["chunk_id"]]
            supporting_indices = [citation.get("chunk_index")]

        valid_support = [
            (str(chunk_id), supporting_indices[index] if index < len(supporting_indices) else None)
            for index, chunk_id in enumerate(supporting_ids)
            if (str(chunk_id), str(citation.get("document_id"))) in valid_pairs
        ]
        if not valid_support:
            continue

        citation["supporting_chunk_ids"] = [item[0] for item in valid_support]
        citation["supporting_chunk_indices"] = [item[1] for item in valid_support]
        citation["chunk_id"] = valid_support[0][0]
        citation["chunk_index"] = valid_support[0][1]
        retained.append(citation)
    return retained

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
