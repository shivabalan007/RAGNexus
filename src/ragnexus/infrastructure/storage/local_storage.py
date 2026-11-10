from pathlib import Path
from uuid import uuid4
import hashlib
import os

class LocalDocumentStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_stream(self, stream, *, workspace_id: str, max_bytes: int) -> tuple[Path, str, int]:
        folder = self.root / workspace_id
        folder.mkdir(parents=True, exist_ok=True)
        temp_path = folder / f".{uuid4().hex}.upload"
        digest = hashlib.sha256()
        size = 0
        try:
            with temp_path.open("xb") as target:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError(f"File exceeds maximum upload size of {max_bytes} bytes")
                    digest.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if size == 0:
                raise ValueError("Uploaded file is empty")
            final_path = folder / f"{uuid4().hex}.bin"
            os.replace(temp_path, final_path)
            return final_path, digest.hexdigest(), size
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Invalid storage path")
        return candidate

    def delete(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)
