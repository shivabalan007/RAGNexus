from __future__ import annotations

import re
import hashlib
from collections.abc import Iterable

from ragnexus.ingestion.base import Chunk, LoadedDocument

_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+.+|[A-Z][A-Z0-9 /&()_-]{3,90}:?)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class StructureAwareChunker:
    """Split by headings/paragraphs first, then produce bounded child chunks.

    Each child retains a compact parent section in metadata so retrieval can
    expand context without requiring an additional relational table. Metadata
    remains citation-addressable through the original child chunk ID.
    """

    def __init__(self, chunk_size: int = 1200, parent_max_chars: int = 5000):
        if chunk_size < 200 or parent_max_chars < chunk_size:
            raise ValueError("Require chunk_size >= 200 and parent_max_chars >= chunk_size")
        self.chunk_size = chunk_size
        self.parent_max_chars = parent_max_chars

    @staticmethod
    def _sections(doc: LoadedDocument) -> list[tuple[str, str]]:
        lines = doc.text.replace("\r\n", "\n").split("\n")
        sections: list[tuple[str, list[str]]] = []
        title = str(doc.metadata.get("section_title") or doc.metadata.get("title") or "Document section")
        current_title = title
        current: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _HEADING.match(stripped) and len(stripped) < 120:
                if current:
                    sections.append((current_title, current))
                current_title = stripped.lstrip("#").strip().rstrip(":") or title
                current = []
            else:
                current.append(line)
        if current:
            sections.append((current_title, current))
        if not sections:
            sections = [(title, lines)]
        return [(heading, "\n".join(body).strip()) for heading, body in sections if "\n".join(body).strip()]

    @staticmethod
    def _split_oversized_unit(text: str, limit: int) -> list[str]:
        """Split long sentences on word boundaries without silently dropping tails."""
        text = text.strip()
        if len(text) <= limit:
            return [text] if text else []
        words = text.split()
        pieces: list[str] = []
        current = ""
        for word in words:
            if len(word) > limit:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(word[i:i + limit] for i in range(0, len(word), limit))
                continue
            candidate = f"{current} {word}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = word
        if current:
            pieces.append(current)
        return pieces

    def _child_pieces(self, parent: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", parent) if p.strip()]
        if not paragraphs:
            paragraphs = [parent.strip()]
        pieces: list[str] = []
        current = ""
        for paragraph in paragraphs:
            sentences = _SENTENCE_BOUNDARY.split(paragraph) if len(paragraph) > self.chunk_size else [paragraph]
            units: list[str] = []
            for sentence in sentences:
                units.extend(self._split_oversized_unit(sentence, self.chunk_size))
            for unit in units:
                candidate = f"{current}\n\n{unit}".strip() if current else unit
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)
                    current = unit
        if current:
            pieces.append(current)
        return pieces

    def split(self, documents: list[LoadedDocument]) -> list[Chunk]:
        output: list[Chunk] = []
        index = 0
        for doc_index, doc in enumerate(documents):
            for section_index, (heading, section_text) in enumerate(self._sections(doc)):
                # Very large sections are subdivided into independent parent windows.
                parent_windows: list[str] = []
                current_parent = ""
                for paragraph in [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]:
                    paragraph_parts = self._split_oversized_unit(paragraph, self.parent_max_chars)
                    for part in paragraph_parts:
                        candidate = f"{current_parent}\n\n{part}".strip() if current_parent else part
                        if len(candidate) <= self.parent_max_chars:
                            current_parent = candidate
                        else:
                            if current_parent:
                                parent_windows.append(current_parent)
                            current_parent = part
                if current_parent:
                    parent_windows.append(current_parent)
                for window_index, parent_text in enumerate(parent_windows):
                    parent_id = hashlib.sha1(
                        f"{doc_index}:{section_index}:{window_index}:{heading}".encode("utf-8")
                    ).hexdigest()[:20]
                    for child_index, piece in enumerate(self._child_pieces(parent_text)):
                        child_text = f"{heading}\n{piece}".strip()
                        metadata = {
                            **doc.metadata,
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "parent_content": parent_text[:self.parent_max_chars],
                            "section_title": heading,
                            "section_path": [heading],
                            "section_index": section_index,
                            "parent_window_index": window_index,
                            "child_index": child_index,
                        }
                        output.append(Chunk(child_text, index, metadata))
                        index += 1
        return output
