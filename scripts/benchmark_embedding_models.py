from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragnexus.config import get_settings
from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.metrics import retrieval_metrics
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from sqlalchemy import select


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline embedding-model comparison against one workspace corpus.")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--dataset", default="evals/datasets/ground_truth.jsonl")
    parser.add_argument("--models", default="sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,intfloat/e5-base-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="evals/reports/embedding_models.json")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    import numpy as np

    settings = get_settings()
    cases = load_jsonl(ROOT / args.dataset)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        with factory() as db:
            membership = db.get(WorkspaceMember, (UUID(args.workspace_id), UUID(args.user_id)))
            if membership is None:
                raise SystemExit("User is not a member of the requested workspace.")
            rows = db.execute(select(DocumentChunk.id, DocumentChunk.content, Document.filename)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(Document.workspace_id == UUID(args.workspace_id), Document.status == "indexed",
                       Document.document_metadata["is_current"].as_boolean().is_(True))
                .order_by(DocumentChunk.id).limit(50000)).all()
        if not rows:
            raise SystemExit("No indexed current chunks were found in this workspace.")
        ids = [str(r.id) for r in rows]
        texts = [r.content for r in rows]
        filenames = [r.filename for r in rows]
        output = {"workspace_id": args.workspace_id, "chunk_count": len(rows), "models": {}}
        for case in cases:
            if case.workspace_id and case.workspace_id != args.workspace_id:
                raise SystemExit(
                    f"Case {case.case_id} targets workspace {case.workspace_id}, but this script loads only "
                    f"--workspace-id {args.workspace_id}. Run a separate benchmark per workspace."
                )

        for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
            import time
            model_started = time.perf_counter()
            model = SentenceTransformer(model_name)
            if model_name.startswith("intfloat/e5-"):
                corpus_inputs = [f"passage: {text}" for text in texts]
                query_prefix = "query: "
            elif model_name.startswith("BAAI/bge-"):
                corpus_inputs = texts
                query_prefix = "Represent this sentence for searching relevant passages: "
            else:
                corpus_inputs = texts
                query_prefix = ""
            corpus_vectors = model.encode(corpus_inputs, normalize_embeddings=True, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
            index_build_ms = (time.perf_counter() - model_started) * 1000
            precisions, recalls, hit_rates, mrrs, ndcgs, latencies = [], [], [], [], [], []
            for case in cases:
                started = time.perf_counter()
                qvec = model.encode([query_prefix + case.query], normalize_embeddings=True, convert_to_numpy=True)[0]
                scores = np.asarray(corpus_vectors) @ qvec
                ranked_indices = np.argsort(-scores)[:args.top_k]
                latency = (time.perf_counter() - started) * 1000
                ranked_ids = [ids[i] for i in ranked_indices]
                relevant = list(case.relevant_chunk_ids)
                if not relevant and case.relevant_source_names:
                    relevant = [ids[i] for i, name in enumerate(filenames) if name in case.relevant_source_names]
                metrics = retrieval_metrics(relevant, ranked_ids, args.top_k)
                precisions.append(metrics["precision_at_k"]); recalls.append(metrics["recall_at_k"])
                hit_rates.append(metrics["hit_rate"]); mrrs.append(metrics["mrr"])
                ndcgs.append(metrics["ndcg"]); latencies.append(latency)
            output["models"][model_name] = {
                "dimension": int(corpus_vectors.shape[1]),
                "precision_at_k": float(np.mean(precisions)),
                "recall_at_k": float(np.mean(recalls)),
                "hit_rate": float(np.mean(hit_rates)),
                "mrr": float(np.mean(mrrs)), "ndcg": float(np.mean(ndcgs)),
                "index_build_ms_including_model_load": float(index_build_ms),
                "query_encode_and_scoring_mean_ms": float(np.mean(latencies)),
                "query_encode_and_scoring_p95_ms": float(np.percentile(latencies, 95)),
            }
            print(model_name, output["models"][model_name])
        dest = ROOT / args.output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
