from pathlib import Path
from docx import Document as DocxDocument
from ragnexus.ingestion.base import LoadedDocument

class DOCXLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        doc = DocxDocument(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table_no, table in enumerate(doc.tables, start=1):
            for row_no, row in enumerate(table.rows, start=1):
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    parts.append(f"[Table {table_no}, row {row_no}] " + " | ".join(cells))
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("No text found in DOCX.")
        return [LoadedDocument(text, {"source_filename": original_filename, "file_type": "docx"})]
