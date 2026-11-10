
from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import create_engine, pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.base import Base
from ragnexus.infrastructure.database.connection import build_database_url
from ragnexus.infrastructure.database import models  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    settings = get_settings()
    url = build_database_url(settings)

    # Do not replace '%' with '%%' here.
    return url.render_as_string(hide_password=False)


def include_name(name, type_, parent_names):
    if type_ == "schema":
        return name == "ragnexus"
    return True


def run_migrations_offline():
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    # Use the same SQLAlchemy URL object that passed your connection test.
    engine = create_engine(
        build_database_url(get_settings()),
        poolclass=pool.NullPool,
    )

    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
                include_schemas=True,
                include_name=include_name,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
