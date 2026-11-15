# Upgrade Notes — Advanced Engineering Package

## What changed

- Added structure-aware chunking and Markdown file support.
- Added bounded multi-query retrieval, parent context expansion, extractive contextual compression and evidence sufficiency diagnostics to the active shared retrieval service.
- Added Dense, BM25 and Hybrid retrieval modes for experiments.
- Added LangGraph Agentic RAG using the same authorized retrieval service, optional Tavily evidence and bounded corrective retries.
- Added persistent human-approval request/decision/consumption infrastructure for future sensitive tools.
- Added 120-row draft evaluation template, evaluation runner/metrics and embedding/reranker experiment scripts.
- Added JSON structured logging, request/trace IDs, optional OpenTelemetry tracing and Redis query-embedding cache.
- Retained the local/S3 storage adapter and existing legacy code as reference.
- Added worker lease ownership checks and stale-job recovery.
- Changed active document content-hash uniqueness so a soft-deleted file can be uploaded again; active duplicate contents remain prevented.

## Required upgrade steps

1. Back up the PostgreSQL database and current document storage folder.
2. Keep your current `.env`; do not replace it with `.env.example`.
3. Install updated dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Review `.env.example` and add only the new optional settings you intend to enable. Keep your current secrets and database URL.
5. Apply the migration:

   ```bash
   PYTHONPATH=src alembic upgrade head
   ```

6. Restart the FastAPI process and the ingestion worker.
7. Run unit tests and smoke-test login, workspace access, document upload/index, document deletion/re-upload, chat/history, standard mode, then agentic mode with Tavily enabled.

The migration `a7b4c2d9f610` targets the `ragnexus` schema and:

- replaces the all-row content-hash unique constraint with a partial unique index over non-deleted rows;
- adds worker lease/heartbeat columns;
- creates the agent approval request table with single-use consumption metadata.

Do not run the migration against a production database without a backup and a reviewed maintenance window. The build environment did not connect to your database; the migration has not been executed against your live database.

## Existing environment and database

The package intentionally excludes `.env`, `.venv`, `.git`, uploaded runtime documents and local caches. Your local `.env`, DB, stored documents and virtual environment should remain untouched. This is a code package; it does not modify your running installation automatically.

## Embedding model constraint

The current active pgvector column is 384 dimensions. The embedding benchmark compares alternative models in memory. Models such as BGE-base and E5-base generally return a different vector dimension; switching the production model requires a separate vector-column migration and re-embedding every indexed document. Do not change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION` to a candidate without that plan.

## Validation performed in the package workspace

- Python compileall: passed.
- Unit tests: 37 passed.
- ZIP archive content and secret/sample-data exclusion: checked at packaging time.

Not validated in this build environment: live PostgreSQL migration, provider requests, live LangGraph execution, Tavily web search, Redis connectivity, AWS S3 operations, OTLP collector export and full browser UI flows. Those require your local configuration and services.
