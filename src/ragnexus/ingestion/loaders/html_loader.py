from pathlib import Path
from bs4 import BeautifulSoup
from ragnexus.ingestion.base import LoadedDocument

class HTMLLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        text = soup.get_text("\n", strip=True)
        if not text:
            raise ValueError("No readable text found in HTML.")
        return [LoadedDocument(text, {"source_filename": original_filename, "file_type": "html", "title": soup.title.get_text(strip=True) if soup.title else None})]
