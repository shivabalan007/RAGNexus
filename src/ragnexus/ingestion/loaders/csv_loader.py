from pathlib import Path
import csv
from ragnexus.ingestion.base import LoadedDocument

class CSVLoader:
    def load(self, path: Path, *, original_filename: str) -> list[LoadedDocument]:
        result: list[LoadedDocument] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV must contain a header row.")
            for row_no, row in enumerate(reader, start=1):
                text = " | ".join(f"{key}: {value or ''}" for key, value in row.items())
                if text.strip(" |:"):
                    result.append(LoadedDocument(text, {
                        "source_filename": original_filename,
                        "file_type": "csv",
                        "row_number": row_no,
                    }))
        if not result:
            raise ValueError("CSV contains no data rows.")
        return result


"""
Reads .csv files and converts each row into a Document object. Useful for structured data like FAQs or knowledge bases.
"""