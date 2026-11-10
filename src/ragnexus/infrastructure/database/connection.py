
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import sessionmaker

from ragnexus.config import Settings


def build_database_url(settings: Settings) -> URL:
    raw_url = settings.DATABASE_URL.get_secret_value()

    url = make_url(raw_url)

    if url.drivername != "postgresql+psycopg2":
        raise ValueError(
            "DATABASE_URL must use postgresql+psycopg2"
        )

    if not url.password:
        raise ValueError("Database password is missing")

    return url


def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        build_database_url(settings),
        pool_pre_ping=True,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        echo=settings.DB_ECHO,
    )


def create_session_factory(engine: Engine):
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
