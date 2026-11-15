# RAGNexus deployment hardening checklist

This checklist is a handoff guide. It does not mean the package has been deployed or that external services have been validated.

## Before production

- [ ] Use Python 3.11 and install `requirements.txt` in a clean virtual environment.
- [ ] Set `APP_ENV=production` and `DEBUG=false`.
- [ ] Generate a unique `JWT_SECRET_KEY` of at least 32 characters and a valid dummy password hash.
- [ ] Keep `.env` outside source control and restrict filesystem permissions.
- [ ] Use a dedicated PostgreSQL role with only the permissions required by the application.
- [ ] Back up PostgreSQL and the original document storage before migrations or deployment.
- [ ] Review `alembic current`; run `PYTHONPATH=src alembic upgrade head` only after verifying the expected predecessor revision.
- [ ] Verify restore from a backup, not only backup creation.
- [ ] Configure TLS at the reverse proxy and restrict inbound network access to the API, PostgreSQL, Redis and OTLP collector.
- [ ] Set explicit CORS origins at the deployment boundary; do not use wildcard origins with credentials.
- [ ] Use a managed secret store or protected environment injection for OpenRouter/Groq, Tavily, AWS and database credentials.

## Storage and cache

- [ ] Keep `STORAGE_BACKEND=local` for local development.
- [ ] For S3, set `STORAGE_BACKEND=s3`, `S3_BUCKET`, `AWS_REGION` and a least-privilege IAM role/policy.
- [ ] Verify S3 upload, download, delete, object-key isolation, encryption, lifecycle and recovery before moving production documents.
- [ ] Do not store AWS access keys in source files or the ZIP.
- [ ] Leave `REDIS_URL` empty to disable caching. When enabled, use private networking and authentication/TLS supported by the Redis deployment.
- [ ] Verify cache namespace/TTL and invalidation behaviour against document updates, deletion, workspace scope and model changes.

## Observability and operations

- [ ] Keep `OTEL_ENABLED=false` until an OTLP collector endpoint is configured.
- [ ] When enabled, set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME` and verify traces arrive at the collector.
- [ ] Collect JSON application and worker logs; restrict access and define retention.
- [ ] Alert on API 5xx rate, ingestion failures/stale jobs, queue age, database saturation, provider errors and p95 latency.
- [ ] Set provider budgets/rate limits outside the application and monitor token usage.
- [ ] Set container/process memory and CPU limits based on measured model/reranker load.
- [ ] Run API, worker and Streamlit as separate supervised processes; configure restart policy and graceful shutdown.
- [ ] Verify health endpoints and a full upload → index → retrieve → delete lifecycle after deployment.

## Known scope boundaries

- Redis caching is limited to query embeddings; it is optional and PostgreSQL remains authoritative.
- Tavily is read-only web evidence. Web page content is untrusted and must not be treated as tool instructions.
- Human approval request/decision/consumption APIs are included for sensitive operations. No destructive agent tool is registered by default; a future mutating tool must consume the exact one-time approval and independently enforce ordinary workspace permissions immediately before its side effect.
- No rate-limiting middleware or production reverse proxy is bundled. Add and test those at the deployment boundary.
- The supplied 120-case evaluation file is a draft labeling template, not a completed ground-truth dataset. Populate and review it against your own corpus before publishing results.
