from pathlib import Path
from ragnexus.ingestion.base import LoadedDocument

class TextLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("The text file is empty.")
        suffix = Path(original_filename).suffix.lower().lstrip(".") or "txt"
        file_type = "markdown" if suffix == "md" else suffix
        return [LoadedDocument(text, {"source_filename": original_filename, "file_type": file_type})]


"""
Reads .txt files from disk and returns list of Document objects. Sets source filename in metadata for citation tracking.
"""