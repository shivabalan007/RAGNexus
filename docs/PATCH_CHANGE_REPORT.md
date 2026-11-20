# RAGNexus patch change report

## Scope

This patch is based on the uploaded `rag1911.zip`. It extends the current Standard RAG and Agentic RAG architecture. It does not replace the provider abstraction, PostgreSQL, pgvector, Celery, the ingestion pipeline, workspace authorization, document scoping, dense/BM25/RRF retrieval, reranking, or citation validation. No existing files are deleted and no database migration is required for these changes.

The ZIP contains only changed and added files, with paths relative to the project root.

## Implemented changes

### Intent, conversation and pipeline selection
- Wire a shared low-cost `memory / knowledge / general` intent router into both selected chat modes.
- Preserve knowledge follow-up rewriting, but skip an extra rewrite-model call for standalone questions.
- Use PostgreSQL as durable conversation history and a user/session-scoped Redis recent-history cache.
- Preserve same-session switching between Standard and Agentic; persist the pipeline used on each user/assistant message.

### Tavily and Agentic routing
- Unwrap the configured `SecretStr` Tavily API key before constructing the client.
- Route explicit web requests and freshness-sensitive queries to the existing Agentic hybrid source plan when the existing web-search opt-in is enabled.
- Use Tavily after configured corrective retries when document evidence remains insufficient.
- Add one final web fallback when generation abstains and Tavily has not already been attempted. This prevents an optimistic retrieval grader from permanently suppressing web fallback.
- Keep web search disabled when the per-request opt-in is false.
- Expose attempted/status/reason/result count/cache hit/latency/error diagnostics; the UI also indicates when Tavily was not selected.
- Cache normalized Tavily results in Redis with shorter TTL for freshness-sensitive queries.

### Redis caching
- Add optional fail-open Redis JSON caching. Redis is not required for chat to run.
- Wire query-embedding caching, with JSON-native float vectors and model/dimension-aware keys.
- Wire authorization-checked retrieval-result caching. Cache payloads store chunk IDs/scores; chunk content is reloaded from the current authorized database scope.
- Wire recent conversation-context caching.
- Add conservative exact-repeat answer caching scoped to user, session, workspace, document scope/index fingerprint, pipeline, provider/model and recent context. Only successful evidence-backed responses are cached.
- Keep cache DB 0 separate from Celery broker/results DB 1/2 as documented.

Note: conversation-history caching saves repeated PostgreSQL history reads; it does not automatically reduce prompt tokens because the relevant context is still sent to the model. Exact-repeat answer caching and cached Tavily/retrieval operations avoid repeat work.

### Documents, metrics and UI
- Preserve the existing `file_size_bytes` API field for compatibility, while adding actual indexed chunk count and embedding model/dimension.
- Display indexed chunks and embedding details in the Streamlit document list instead of file size.
- Render compact per-message metrics in seconds, identify the selected pipeline on each assistant response, and show retrieval/cache/Tavily diagnostics.
- Add Agentic stage timing and generation timing; missing generation time remains absent when no generation call occurred.
- Reuse a Streamlit-session HTTP client and remove an unnecessary explicit rerun after a chat response. Browser focus/scroll performance still requires verification on the user's Windows machine.

### Correctness fixes found during the final audit
- Import SQLAlchemy `func` in the document listing endpoint; the indexed chunk-count query uses `func.count()`.
- Preserve query casing in embedding/retrieval/answer cache keys so case-sensitive technical identifiers are not conflated.
- Include embedding dimension in query-embedding cache identity.
- Permit Agentic web-only operation if a selected-document session has no remaining selected documents and the user has enabled web search.
- Correct the deployment checklist's evaluation-dataset description to match the 100-case dataset.

## Changed files (19)

- `docs/DEPLOYMENT_CHECKLIST.md`
- `src/ragnexus/agentic/graph.py`
- `src/ragnexus/agentic/tools/tavily_search.py`
- `src/ragnexus/api/v1/endpoints/chat_router_factory.py`
- `src/ragnexus/api/v1/endpoints/ingestion.py`
- `src/ragnexus/api/v1/orchestration_proxy.py`
- `src/ragnexus/config.py`
- `src/ragnexus/infrastructure/cache/redis_cache.py`
- `src/ragnexus/rag/orchestration/bootstrap.py`
- `src/ragnexus/rag/orchestration/followup.py`
- `src/ragnexus/rag/orchestration/prompts.py`
- `src/ragnexus/rag/orchestration/service.py`
- `src/ragnexus/rag/routing/intent_router.py`
- `src/ragnexus/rag/routing/router.py`
- `src/ragnexus/retrieval/result.py`
- `src/ragnexus/retrieval/service.py`
- `tests/unit/test_evaluation_expanded.py`
- `tests/unit/test_runtime_wiring.py`
- `ui/app.py`

## Added files (2)

- `docs/ROUTING_CACHE_AND_UI_UPDATE.md`
- `tests/unit/test_routing_and_caching.py`

## Automated verification performed here

Environment: Python 3.13.5. The project targets Python 3.11.

Commands:

```bash
export PYTHONPATH=src
export JWT_SECRET_KEY=test-secret-key-for-test-only
export DUMMY_PASSWORD_HASH=test-hash
export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/ragnexus
python -m compileall -q src workers scripts tests ui
pytest -q -rs
```

Result: **66 passed, 4 skipped**. All four skips are due to `pgvector` not being installed in this isolated environment. The compile check passed.

## Not verified in this environment

- Actual graph compilation/execution with the installed LangGraph dependency.
- Real PostgreSQL and pgvector query execution, migrations and document-list endpoint integration.
- Redis/Memurai connectivity, TTLs and fail-open behaviour against a live Redis server.
- Live Tavily network requests and credentials.
- Live OpenRouter and Ollama requests. The latest Ollama fix in the uploaded base was preserved; provider code was not changed by this patch.
- Browser typing/focus/scroll behaviour and UI responsiveness on the user's Windows 10 machine.
- Docker Compose startup and AWS deployment.

These need the local smoke tests in `ROUTING_CACHE_AND_UI_UPDATE.md` and the user's existing deployment checklist. This patch is not a claim that those external integration tests have passed.
