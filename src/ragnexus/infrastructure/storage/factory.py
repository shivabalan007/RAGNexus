from ragnexus.config import Settings
from ragnexus.infrastructure.storage.local_storage import LocalDocumentStorage


def build_document_storage(settings: Settings):
    backend = getattr(settings, "STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalDocumentStorage(settings.DOCUMENT_STORAGE_DIR)
    if backend == "s3":
        from ragnexus.infrastructure.storage.s3_storage import S3DocumentStorage
        return S3DocumentStorage(
            bucket=settings.S3_BUCKET,
            region=settings.AWS_REGION,
            prefix=settings.S3_PREFIX,
        )
    raise ValueError(f"Unsupported STORAGE_BACKEND: {backend}")
