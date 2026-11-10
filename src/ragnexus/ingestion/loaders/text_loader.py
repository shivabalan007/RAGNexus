from pathlib import Path
from ragnexus.ingestion.base import LoadedDocument

class TextLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("The text file is empty.")
        return [LoadedDocument(text, {"source_filename": original_filename, "file_type": "txt"})]


"""
Reads .txt files from disk and returns list of Document objects. Sets source filename in metadata for citation tracking.
"""