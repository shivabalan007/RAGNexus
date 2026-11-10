from pathlib import Path
from pypdf import PdfReader
from ragnexus.ingestion.base import LoadedDocument

class PDFLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        reader = PdfReader(str(path), strict=False)
        result: list[LoadedDocument] = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                result.append(LoadedDocument(text, {
                    "source_filename": original_filename,
                    "file_type": "pdf",
                    "page_number": page_no,
                }))
        if not result:
            raise ValueError("No extractable text found in PDF. It may be scanned; OCR is not enabled in this build.")
        return result


"""
Reads .pdf files using a PDF library and extracts raw text per page. Returns list of Document objects with page number in metadata.
"""