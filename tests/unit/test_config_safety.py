import pytest
from pydantic import ValidationError

from ragnexus.config import Settings


def _settings(**overrides):
    values = {
        "_env_file": None,
        "JWT_SECRET_KEY": "development-secret",
        "DUMMY_PASSWORD_HASH": "test-password-hash",
        "DATABASE_URL": "postgresql+psycopg2://user:password@localhost:5432/rag_memory",
    }
    values.update(overrides)
    return Settings(**values)


def test_production_rejects_debug_mode():
    with pytest.raises(ValidationError, match="DEBUG must be false"):
        _settings(APP_ENV="production", DEBUG=True, JWT_SECRET_KEY="x" * 48)


def test_production_requires_long_jwt_secret():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _settings(APP_ENV="production", DEBUG=False, JWT_SECRET_KEY="short-secret")


def test_s3_storage_requires_bucket():
    with pytest.raises(ValidationError, match="S3_BUCKET is required"):
        _settings(STORAGE_BACKEND="s3", S3_BUCKET=None)
