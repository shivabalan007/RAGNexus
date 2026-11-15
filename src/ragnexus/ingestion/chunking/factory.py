from ragnexus.ingestion.chunking.fixed_chunker import FixedChunker
from ragnexus.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from ragnexus.ingestion.chunking.semantic_chunker import SemanticChunker
from ragnexus.ingestion.chunking.structure_aware import StructureAwareChunker

def build_chunker(strategy: str, *, embedder=None):
    if strategy == "fixed":
        return FixedChunker()
    if strategy == "sliding_window":
        return SlidingWindowChunker()
    if strategy == "semantic":
        return SemanticChunker(embedder=embedder)
    if strategy == "structure_aware":
        from ragnexus.config import get_settings
        settings = get_settings()
        return StructureAwareChunker(settings.STRUCTURE_CHUNK_SIZE, settings.STRUCTURE_PARENT_MAX_CHARS)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
