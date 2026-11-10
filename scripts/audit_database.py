from sqlalchemy import inspect, text

from ragnexus.config import settings
from ragnexus.infrastructure.database.connection import (
    create_database_engine,
)


def audit_database():
    engine = create_database_engine(settings)

    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(
                    text("SET TRANSACTION READ ONLY")
                )

                inspector = inspect(connection)

                print("\n========== DATABASE AUDIT ==========")

                result = connection.execute(
                    text("""
                        SELECT current_database(),
                               current_user,
                               current_schema()
                    """)
                ).one()

                print(f"Database: {result[0]}")
                print(f"User: {result[1]}")
                print(f"Schema: {result[2]}")

                extensions = connection.execute(
                    text("""
                        SELECT extname
                        FROM pg_extension
                        ORDER BY extname
                    """)
                ).scalars().all()

                print("\nExtensions:")
                for extension in extensions:
                    print(f"  - {extension}")

                tables = inspector.get_table_names()

                print("\nTables:")

                for table in tables:
                    print(f"\n--- {table} ---")

                    for column in inspector.get_columns(table):
                        print(
                            f"  COLUMN: {column['name']} "
                            f"| TYPE: {column['type']} "
                            f"| NULLABLE: {column['nullable']}"
                        )

                    for constraint in inspector.get_unique_constraints(table):
                        print(
                            f"  UNIQUE: {constraint['name']} "
                            f"| COLUMNS: {constraint['column_names']}"
                        )

                    for index in inspector.get_indexes(table):
                        print(
                            f"  INDEX: {index['name']} "
                            f"| COLUMNS: {index['column_names']} "
                            f"| UNIQUE: {index['unique']}"
                        )

                    for fk in inspector.get_foreign_keys(table):
                        print(
                            f"  FOREIGN KEY: {fk['constrained_columns']} "
                            f"-> {fk['referred_table']}."
                            f"{fk['referred_columns']}"
                        )

                    preparer = connection.dialect.identifier_preparer
                    quoted_table = preparer.quote(table)

                    count = connection.execute(
                        text(f"SELECT COUNT(*) FROM {quoted_table}")
                    ).scalar_one()

                    print(f"  ROW COUNT: {count}")

                print("\n========== AUDIT COMPLETE ==========")

    finally:
        engine.dispose()


if __name__ == "__main__":
    audit_database()