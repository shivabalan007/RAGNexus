from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "RAGNexus"
    APP_VERSION: str = "0.3.0"
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
    ENABLE_MULTI_QUERY: bool = True
    MULTI_QUERY_MAX_QUERIES: int = Field(default=3, ge=1, le=6)
    ENABLE_PARENT_EXPANSION: bool = True
    ENABLE_CONTEXTUAL_COMPRESSION: bool = True
    ENABLE_EVIDENCE_SUFFICIENCY: bool = True
    EVIDENCE_MIN_TOKEN_OVERLAP: float = Field(default=0.30, ge=0.0, le=1.0)
    EVIDENCE_MIN_DENSE_SIMILARITY: float = Field(default=0.42, ge=-1.0, le=1.0)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CHUNKING_STRATEGY: Literal["fixed", "sliding_window", "semantic", "structure_aware"] = "structure_aware"
    STRUCTURE_CHUNK_SIZE: int = Field(default=1200, ge=200, le=5000)
    STRUCTURE_PARENT_MAX_CHARS: int = Field(default=5000, ge=500, le=20000)

    # Agentic RAG
    AGENTIC_MAX_RETRIES: int = Field(default=2, ge=0, le=3)
    AGENTIC_ENABLE_QUERY_ROUTING: bool = True
    AGENTIC_ENABLE_RETRIEVAL_GRADING: bool = True
    AGENTIC_ENABLE_ANSWER_VERIFICATION: bool = True
    TAVILY_MAX_RESULTS: int = Field(default=5, ge=1, le=10)
    TAVILY_SEARCH_DEPTH: Literal["basic", "advanced"] = "basic"

    # Worker lease / retry controls
    INGESTION_LEASE_SECONDS: int = Field(default=120, ge=30, le=3600)
    INGESTION_RECOVERY_INTERVAL_SECONDS: int = Field(default=15, ge=5, le=300)
    INGESTION_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=10)

    # Observability
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_SERVICE_NAME: str = "ragnexus"

    DOCUMENT_STORAGE_DIR: str = "./data/documents"
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    S3_BUCKET: str | None = None
    AWS_REGION: str | None = None
    S3_PREFIX: str = "ragnexus"
    MAX_UPLOAD_SIZE_MB: int = Field(default=25, ge=1, le=200)

    LLM_PROVIDER: Literal["openrouter", "groq"] = "openrouter"
    LLM_MODEL: str | None = None

    @model_validator(mode="after")
    def validate_deployment_safety(self):
        if self.APP_ENV == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be false when APP_ENV=production")
            if len(self.JWT_SECRET_KEY.get_secret_value()) < 32:
                raise ValueError("JWT_SECRET_KEY must contain at least 32 characters in production")
        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()