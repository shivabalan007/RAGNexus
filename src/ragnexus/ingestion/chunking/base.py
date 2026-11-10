from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int
    overlap: int

"""
Defines ChunkConfig dataclass with chunk_size and overlap parameters. Shared config used across different chunking strategies.
"""