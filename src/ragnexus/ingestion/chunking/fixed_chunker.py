from ragnexus.ingestion.base import Chunk, LoadedDocument

class FixedChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 180):
        if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size >= 100 and 0 <= overlap < chunk_size")
        self.chunk_size, self.overlap = chunk_size, overlap

    def split(self, documents: list[LoadedDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for doc in documents:
            text = doc.text
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if end < len(text):
                    boundary = text.rfind(" ", start, end)
                    if boundary > start + self.chunk_size // 2:
                        end = boundary
                piece = text[start:end].strip()
                if piece:
                    chunks.append(Chunk(piece, index, {**doc.metadata, "char_start": start, "char_end": end}))
                    index += 1
                if end >= len(text):
                    break
                start = max(start + 1, end - self.overlap)
        return chunks


"""
Splits document into fixed-size character chunks with overlap. Simple baseline chunker — used in early version.
"""