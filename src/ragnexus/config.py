from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "RAGNexus"
    APP_VERSION: str = "0.1.0"
    APP_ENV: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"

    # Authentication and security
    JWT_SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str = "ragnexus"
    JWT_AUDIENCE: str = "ragnexus-api"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=15,
        ge=5,
        le=60,
    )

    # Dummy password hash for unknown-user login attempts
    DUMMY_PASSWORD_HASH: str

    # PostgreSQL
    # DB_HOST: str | None = None
    # DB_PORT: int = 5432
    # DB_NAME: str | None = None
    # DB_USER: str | None = None
    # DB_PASSWORD: SecretStr | None = None

    DATABASE_URL: SecretStr

    DB_POOL_SIZE: int = Field(default=10, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DB_POOL_TIMEOUT: int = Field(default=30, ge=1)
    DB_ECHO: bool = False

    # Redis
    REDIS_URL: str | None = None
    REDIS_TTL_SECONDS: int = Field(default=1800, ge=60)

    # LLM providers
    OPENROUTER_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    TAVILY_API_KEY: SecretStr | None = None

    # Embeddings and retrieval
    EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = 384
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 3

    DOCUMENT_STORAGE_DIR: str = "./data/documents"
    MAX_UPLOAD_SIZE_MB: int = Field(default=25, ge=1, le=200)

    LLM_PROVIDER: Literal["openrouter", "groq"] = "openrouter"
    LLM_MODEL: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()