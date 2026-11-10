from ragnexus.ingestion.chunking.fixed_chunker import FixedChunker
from ragnexus.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from ragnexus.ingestion.chunking.semantic_chunker import SemanticChunker

def build_chunker(strategy: str, *, embedder=None):
    if strategy == "fixed":
        return FixedChunker()
    if strategy == "sliding_window":
        return SlidingWindowChunker()
    if strategy == "semantic":
        return SemanticChunker(embedder=embedder)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
