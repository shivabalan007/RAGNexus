# RAGNexus Retrieval and Answer Evaluation

## Ground-truth dataset

`evals/datasets/ground_truth_120.template.jsonl` is a draft questionnaire template, not a labeled benchmark. The 120 placeholder rows are ignored by `load_jsonl` while their status is `draft`.

Create your working dataset:

```bash
cp evals/datasets/ground_truth_120.template.jsonl evals/datasets/ground_truth.jsonl
```

For each row:

- Replace placeholders with a real question from your actual corpus.
- Set `status` to `ready` only after human review.
- Provide a reference answer and expected facts.
- Add relevant chunk UUIDs or relevant source filenames.
- Use `should_abstain: true` only when the permitted corpus genuinely lacks the answer.
- Add workspace and document IDs where needed for scope tests.
- Do not put personal/private corpus material in public portfolio artifacts without permission.

Validate before running an experiment:

```bash
PYTHONPATH=src python scripts/validate_eval_dataset.py --dataset evals/datasets/ground_truth.jsonl
```

Use `relevant_chunk_ids` for precise ranking evaluation. Chunk IDs change when documents are re-ingested or versioned. Use source names for document-level evaluation when a less granular but more stable judgment is preferred.

## Retrieval configuration matrix

```bash
PYTHONPATH=src python scripts/run_retrieval_benchmark.py \
  --workspace-id YOUR_WORKSPACE_UUID \
  --user-id YOUR_USER_UUID
```

The matrix compares dense, BM25, hybrid RRF, hybrid with reranking, query rewrite, multi-query, and the full advanced pipeline. The query-rewrite and multi-query configurations call your configured LLM provider even without `--generate`, so the matrix requires a provider key and incurs those model costs. Metrics include Recall@K, Precision@K, hit rate, MRR and nDCG, plus p50/p95/mean latency and token usage where generation was enabled.

To additionally include answer generation, answer correctness, citations and an LLM-as-judge faithfulness signal (these extra calls also incur provider cost):

```bash
PYTHONPATH=src python scripts/run_retrieval_benchmark.py \
  --workspace-id YOUR_WORKSPACE_UUID \
  --user-id YOUR_USER_UUID \
  --generate
```

Answer metrics include normalized exact match, token F1, lexical expected-fact coverage, citation precision/coverage and verifier faithfulness. Lexical expected-fact coverage is not a semantic judge; inspect per-case results and manually review a sample.

## Embedding model comparison

```bash
PYTHONPATH=src python scripts/benchmark_embedding_models.py \
  --workspace-id YOUR_WORKSPACE_UUID \
  --user-id YOUR_USER_UUID
```

This script loads the current indexed workspace chunk corpus, builds model-specific vectors in memory, and ranks each query. E5 and BGE query/document prefixes are applied. The index-build metric includes model loading and corpus encoding. This is an offline comparison; it does not replace stored pgvector embeddings.

## Reranker comparison

```bash
PYTHONPATH=src python scripts/benchmark_rerankers.py \
  --workspace-id YOUR_WORKSPACE_UUID \
  --user-id YOUR_USER_UUID
```

The script obtains the same hybrid candidate set for each case, loads each CrossEncoder candidate, and compares Recall@K, MRR, nDCG, model-load time and reranking latency.

## Interpreting results

- Keep the dataset, workspace corpus, filters, candidate count and final K constant when comparing configurations.
- Run each benchmark more than once if model download, cold-start or hardware variance affects timings.
- Separate retrieval-only latency from generated-answer latency when interpreting results.
- Compare quality, p95 latency, memory and operational cost together.
- Do not replace the production embedding model solely because a model performs better on a small hand-picked sample.
- Use a representative, manually labeled 100–200 question set. Do not treat the 120-row draft template as proof of quality.
