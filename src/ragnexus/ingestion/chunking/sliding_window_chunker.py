from ragnexus.ingestion.base import Chunk, LoadedDocument

class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 250):
        if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size >= 100 and 0 <= overlap < chunk_size")
        self.chunk_size, self.overlap = chunk_size, overlap

    def split(self, documents: list[LoadedDocument]) -> list[Chunk]:
        output, idx = [], 0
        for doc in documents:
            start = 0
            while start < len(doc.text):
                end = min(start + self.chunk_size, len(doc.text))
                text = doc.text[start:end].strip()
                if text:
                    output.append(Chunk(text, idx, {**doc.metadata, "char_start": start, "char_end": end}))
                    idx += 1
                if end == len(doc.text):
                    break
                start = end - self.overlap
        return output


"""
Takes a text string and splits into overlapping windows of fixed size. Overlap ensures context at chunk boundaries is not lost between adjacent chunks.
"""