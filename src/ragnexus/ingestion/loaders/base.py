from pathlib import Path
from typing import Protocol
from ragnexus.ingestion.base import LoadedDocument

class Loader(Protocol):
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]: ...
