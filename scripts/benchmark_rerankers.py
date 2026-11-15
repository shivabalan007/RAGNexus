from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragnexus.config import get_settings
from ragnexus.embeddings.query_embedder import get_query_embedder
from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.metrics import retrieval_metrics
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.service import RetrievalService


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CrossEncoder reranker models over the same hybrid candidate sets.")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--dataset", default="evals/datasets/ground_truth.jsonl")
    parser.add_argument("--models", default="cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base,BAAI/bge-reranker-large")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="evals/reports/reranker_models.json")
    args = parser.parse_args()
    cases = load_jsonl(ROOT / args.dataset)
    settings = get_settings()
    embedder = get_query_embedder()
    base = RetrievalService(embed_query=embedder.embed_query, embedding_dimension=settings.EMBEDDING_DIMENSION)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    results = {"workspace_id": args.workspace_id, "models": {}}
    try:
        with factory() as db:
            candidates = []
            for case in cases:
                result = base.search(db, user_id=UUID(args.user_id), workspace_id=UUID(case.workspace_id or args.workspace_id),
                                     query=case.query, document_ids=[UUID(x) for x in case.document_ids] or None,
                                     candidate_k=args.candidate_k, final_k=args.candidate_k,
                                     use_query_rewrite=False, use_multi_query=False, use_reranker=False,
                                     use_parent_expansion=False, use_compression=False)
                candidates.append((case, result.chunks))
            for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
                import time
                model_started = time.perf_counter()
                reranker = CrossEncoderReranker(model_name=model_name)
                # Force the lazy model load separately so model startup cost is visible.
                reranker._get_scorer()
                model_load_ms = (time.perf_counter() - model_started) * 1000
                precisions, recalls, hit_rates, mrrs, ndcgs, elapsed = [], [], [], [], [], []
                for case, chunks in candidates:
                    ranked, ms = reranker.rerank(case.query, chunks.copy(), top_k=args.top_k)
                    retrieved_ids = [str(x.chunk_id) for x in ranked]
                    relevant = list(case.relevant_chunk_ids)
                    if not relevant and case.relevant_source_names:
                        relevant = [str(x.chunk_id) for x in chunks if x.metadata.get("filename") in case.relevant_source_names]
                    metrics = retrieval_metrics(relevant, retrieved_ids, args.top_k)
                    precisions.append(metrics["precision_at_k"]); recalls.append(metrics["recall_at_k"])
                    hit_rates.append(metrics["hit_rate"]); mrrs.append(metrics["mrr"])
                    ndcgs.append(metrics["ndcg"]); elapsed.append(ms)
                results["models"][model_name] = {
                    "precision_at_k": sum(precisions)/len(precisions),
                    "recall_at_k": sum(recalls)/len(recalls),
                    "hit_rate": sum(hit_rates)/len(hit_rates),
                    "mrr": sum(mrrs)/len(mrrs), "ndcg": sum(ndcgs)/len(ndcgs),
                    "rerank_mean_ms": sum(elapsed)/len(elapsed),
                    "model_load_ms": model_load_ms}
                print(model_name, results["models"][model_name])
        dest = ROOT / args.output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
