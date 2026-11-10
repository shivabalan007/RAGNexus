
from pathlib import Path

from ragnexus.ingestion.loaders.pdf_loader import PDFLoader
from ragnexus.ingestion.loaders.text_loader import TextLoader
from ragnexus.ingestion.loaders.csv_loader import CSVLoader
from ragnexus.ingestion.loaders.docx_loader import DOCXLoader
from ragnexus.ingestion.loaders.html_loader import HTMLLoader


LOADERS = {
    ".pdf": PDFLoader(),
    ".txt": TextLoader(),
    ".csv": CSVLoader(),
    ".docx": DOCXLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


def load_file(path: Path, *, original_filename: str):
    # The physical storage file uses .bin.
    # Determine the actual document type from the original filename.
    extension = Path(original_filename).suffix.lower()

    loader = LOADERS.get(extension)

    if loader is None:
        raise ValueError(
            f"Unsupported file extension: {extension}. "
            f"Supported: {', '.join(LOADERS.keys())}"
        )

    if not path.exists():
        raise FileNotFoundError(f"Stored document not found: {path}")

    return loader.load(
        path,
        original_filename=original_filename,
    )
