# RAGNexus Advanced Engineering Architecture

## Architectural direction

RAGNexus remains a modular monolith:

- FastAPI owns authentication, workspace authorization, sessions, document APIs and chat APIs.
- PostgreSQL in the `ragnexus` schema remains the system of record; pgvector remains the active vector store.
- The ingestion worker remains a separate process with database-backed leases and bounded retries.
- The standard and agentic chat modes share one `RetrievalService`.
- LangGraph is a control-flow/orchestration layer. It does not own a second copy of retrieval logic.
- Streamlit remains the UI and calls the FastAPI API.

Do not introduce microservices, Kafka, Kubernetes, Qdrant, or another orchestration framework as part of this change. Add them only after measured requirements justify them.

## Standard retrieval path

1. Resolve contextual follow-up questions into standalone queries.
2. Verify workspace membership and selected-document scope before embedding or reading chunks.
3. Optionally rewrite and generate a bounded number of query variants.
4. Run dense pgvector and/or request-scoped BM25 retrieval according to the selected retrieval mode.
5. Fuse multiple rankings with reciprocal rank fusion.
6. Optionally rerank with the configured CrossEncoder.
7. Expand child hits to their parent section context.
8. Optionally apply extractive contextual compression.
9. Check evidence sufficiency before generation.
10. Generate from the retrieved evidence and validate citations against the actual evidence list.

Parent-child data is represented using each child's `chunk_metadata` (`parent_id`, `parent_content`, section and child indices). This avoids adding a separate parent table in this iteration. Citations remain attached to the retrieved child chunk IDs.

## Agentic path

The LangGraph workflow routes the request, reuses the same authorized `RetrievalService`, grades evidence, performs a bounded corrective rewrite when needed, optionally invokes read-only Tavily search when the user enabled web search, generates, and verifies the answer against document and web evidence.

The workflow has a bounded retry count. Tavily is read-only. Retrieved web content and uploaded document text are untrusted evidence, never instructions. Web citations preserve title and URL and use a separate `source_type=web` marker.

### Sensitive actions

The approval API persists proposed actions, requires workspace owners/admins to approve, prevents self-approval, binds consumption to exact canonicalized arguments, rechecks requester membership and approver role, and makes approval single-use with a 24-hour approval window.

No autonomous write-capable agent tools are enabled in this build. Any future mutating tool must independently perform the normal permission check and consume its matching approval immediately before the side effect. An approval is not authorization.

## Chunking and embeddings

The `structure_aware` chunker separates heading/paragraph structure and stores a bounded parent section in child metadata. It preserves oversized text by splitting on word boundaries. Existing fixed, sliding-window and semantic chunkers remain selectable.

The active pgvector column is 384 dimensions. The included BGE-base and E5-base embedding experiments produce 768-dimensional vectors and are offline comparisons only. Do not set one of those models as the production `EMBEDDING_MODEL` without a database vector-dimension migration and a full re-embedding/re-index of documents. Keep the current model and dimension aligned.

## Caching and observability

Redis is optional and currently caches normalized query embeddings keyed by embedding model and SHA-256 query digest. PostgreSQL remains authoritative. Redis failure falls back to uncached computation.

Structured JSON logs avoid prompt/document contents and secrets. OpenTelemetry traces are opt-in through `OTEL_ENABLED` and the OTLP endpoint. Instrumentation failures should not prevent the API from operating.

## Storage

`STORAGE_BACKEND=local` remains the default for development. The existing storage factory supports an S3 adapter behind the same interface. S3 can be configured later without deploying the app to AWS during development. Keep object metadata and lifecycle state in PostgreSQL.
