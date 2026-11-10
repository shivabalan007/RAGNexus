from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class LoadedDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

"""
Defines the Document dataclass with text and metadata fields. Every file loaded in the system becomes a Document object.
"""