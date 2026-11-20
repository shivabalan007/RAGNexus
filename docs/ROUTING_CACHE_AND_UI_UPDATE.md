# RAGNexus routing, cache and chat UI update

This update extends the current architecture. It does not replace Standard RAG, Agentic RAG, the vector store, workspace authorization, document-scope checks, ingestion, or the provider abstraction.

## 1. Intent and source routing

Both chat pipelines now use the shared low-cost `IntentRouter` before retrieval:

- `general`: direct conversational answer; no document retrieval.
- `memory`: answer only from the current authorized conversation history.
- `knowledge`: continue through the selected RAG pipeline.

The router is deterministic for obvious greetings and explicit conversation-recall requests. It does not spend an LLM call on every user message. Ambiguous/factual requests default to `knowledge` so a failed classifier cannot suppress retrieval.

The existing `ReferenceSignalDetector` is also wired into `FollowUpResolver`: standalone knowledge questions bypass the rewrite LLM call, while short follow-ups/referential questions can still be rewritten using recent conversation context.

Agentic RAG retains its separate source planner. Web search remains opt-in via `enable_web_search` (the existing UI labels it Agentic-only):

- Explicit web requests and freshness-sensitive questions select the hybrid source route.
- If the model initially selects documents but document evidence remains insufficient after configured corrective retries, enabled web search is used as the fallback.
- If generation itself abstains for insufficient evidence and web search was enabled but not yet attempted, the graph performs one final Tavily fallback. It never loops back to web search after an attempt.
- A selected-document conversation whose selected documents have since been removed can still use web search when the user enables it; Standard remains scoped to the selected documents.
- With web search disabled, the workflow never calls Tavily.
- Tavily failure, missing configuration, empty results, result count, latency and cache status are exposed in message diagnostics.

The existing Standard RAG pipeline remains document retrieval only. This update does not silently add Tavily to Standard RAG or change its hybrid Dense + BM25 + RRF retrieval.

## 2. Redis responsibilities

PostgreSQL remains the durable source of truth. Redis is optional and fail-open: Redis outages must not prevent chat, retrieval, or ingestion. The Redis cache uses database 0 in the supplied Compose configuration; Celery broker and result backend remain on databases 1 and 2.

| Cache | Stored value | Default TTL |
|---|---|---:|
| Conversation context | Recent role/content messages for the authorized user/session | 30 min |
| Query embedding | JSON-native vector keyed by embedding model and normalized query | 30 min |
| Retrieval | Retrieved chunk IDs, scores and context overrides; text is reloaded from PostgreSQL | 5 min |
| Tavily | Normalized public web results | 15 min |
| Fresh Tavily | Freshness-sensitive web results | 60 sec |
| Exact-repeat answer | Successful evidence-backed answer and citations for the same user/session/context/pipeline/provider/model/index | 5 min |
| Fresh exact-repeat answer | Answer cache for requests with web search enabled or freshness-sensitive wording | 60 sec |

Answer caching is intentionally conservative. It is scoped to the user, session and workspace; includes the pipeline, provider/model, recent context fingerprint, document scope and current corpus fingerprint; and is not used for greetings, memory responses or abstentions. A changed index, model, pipeline or conversation context results in a cache miss. Exact repeats can skip retrieval and generation; semantically similar but differently worded questions are not assumed equivalent.

Conversation-history caching reduces repeated PostgreSQL history reads. It does not, by itself, reduce prompt tokens: the relevant recent history is still supplied to the model. Exact-repeat answer caching is the feature that can avoid a repeat generation call. Search/retrieval caching avoids repeated tool and retrieval work, but a new answer may still require generation.

## 3. Environment configuration

For local Windows/Memurai development, add the following to `.env` (do not commit secrets):

```dotenv
REDIS_CACHE_ENABLED=true
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_TTL_SECONDS=1800
REDIS_CONVERSATION_TTL_SECONDS=1800
REDIS_EMBEDDING_TTL_SECONDS=1800
REDIS_RETRIEVAL_TTL_SECONDS=300
REDIS_TAVILY_TTL_SECONDS=900
REDIS_FRESH_TAVILY_TTL_SECONDS=60
REDIS_ANSWER_TTL_SECONDS=300
REDIS_FRESH_ANSWER_TTL_SECONDS=60
CONVERSATION_HISTORY_LIMIT=20
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=basic
```

Keep Celery on its separate configured URLs, for example:

```dotenv
CELERY_BROKER_URL=redis://127.0.0.1:6379/1
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2
```

The supplied `compose.yaml` already sets `REDIS_URL=redis://redis:6379/0` and Celery URLs on `/1` and `/2` inside Docker. Container service names must not be replaced with `localhost` inside Compose.

To disable the application cache while diagnosing Redis, set `REDIS_CACHE_ENABLED=false` or leave `REDIS_URL` unset. PostgreSQL history and normal retrieval continue to work.

## 4. Document indexing display

The document API retains `file_size_bytes` for compatibility and additionally returns:

- `indexed_chunk_count`: counted from actual chunk rows belonging to the current indexed document.
- `embedding_model` and `embedding_dimension`: the embedding representation recorded at ingestion.

The Streamlit document list displays indexed chunk count and embedding model/dimension instead of file size.

## 5. Chat metrics and pipeline switching

Each assistant message persists its actual `pipeline_mode` and diagnostics. The selector remains enabled during an existing conversation, so each new request can use Standard or Agentic without creating a new session. Both paths share PostgreSQL conversation history and the same per-user/session Redis history cache. Historical assistant responses retain the pipeline used at the time they were generated.

Latency is stored in milliseconds internally and displayed in seconds. Standard metrics include Dense, BM25, RRF and reranking stages. Agentic metrics include routing, retrieval, grading, corrective rewrite, web search, generation and verification. A generation stage is omitted if no LLM generation call took place.

The Streamlit chat request path reuses one HTTP session and writes the new messages into the existing history container without an additional explicit full-page rerun. Browser-side typing/focus behaviour still requires verification in the user's actual browser and machine.

## 6. Local verification

Activate the project's Python 3.11 environment from the project root:

```bash
export PYTHONPATH=src
python -m compileall -q src workers scripts tests ui
pytest -q
```

Useful cache diagnostics (using `redis-cli` or `memurai-cli`):

```bash
redis-cli -n 0 PING
redis-cli -n 0 DBSIZE
redis-cli -n 0 --scan --pattern 'ragnexus:*'
```

Use the Redis CLI only for inspection. Do not run `FLUSHDB` against a Redis instance that also contains Celery or other application data.

## 7. Verification boundaries

Unit tests cover intent classification, web opt-in and fallback routing, Tavily result caching, query-embedding cache serialization, Agentic web-only answer generation and timing, configuration, and existing project tests.

The automated test environment does not provide the user's PostgreSQL data, Windows/Memurai process, browser, OpenRouter/Tavily credentials, or local Ollama models. Therefore this update must still be smoke-tested locally with both providers, both pipeline modes, Redis enabled and disabled, a real Tavily query, and a real upload/index/delete cycle before production deployment.
