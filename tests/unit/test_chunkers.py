from ragnexus.ingestion.base import LoadedDocument
from ragnexus.ingestion.chunking.fixed_chunker import FixedChunker
from ragnexus.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker

def test_fixed_chunker_creates_overlapping_chunks():
    doc = LoadedDocument("A" * 2500, {"source_filename": "sample.txt"})
    chunks = FixedChunker(chunk_size=1000, overlap=100).split([doc])
    assert len(chunks) >= 3
    assert chunks[0].metadata["source_filename"] == "sample.txt"
    assert chunks[0].index == 0

def test_sliding_window_empty_input():
    assert SlidingWindowChunker().split([LoadedDocument("", {})]) == []
