from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragnexus.config import get_settings
from ragnexus.embeddings.query_embedder import get_query_embedder
from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.runner import run_evaluation, write_report, write_markdown_report
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.llm.providers.factory import build_chat_model
from ragnexus.rag.orchestration.prompts import build_rag_messages
from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.retrieval.advanced import MultiQueryGenerator
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.service import RetrievalService
from ragnexus.agentic.verification import AnswerVerifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RAGNexus retrieval configurations on a labeled JSONL dataset.")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--dataset", default="evals/datasets/ground_truth.jsonl")
    parser.add_argument("--output-dir", default="evals/reports/retrieval")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="Also measure answer/citation quality and token usage; incurs LLM cost.")
    args = parser.parse_args()

    cases = load_jsonl(ROOT / args.dataset)
    settings = get_settings()
    model = build_chat_model(settings)
    embedder = get_query_embedder()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    configs = [
        {"name": "dense_baseline", "mode": "dense", "rewrite": False, "multi": False, "rerank": False, "parent": False, "compress": False},
        {"name": "bm25_baseline", "mode": "bm25", "rewrite": False, "multi": False, "rerank": False, "parent": False, "compress": False},
        {"name": "hybrid_rrf", "mode": "hybrid", "rewrite": False, "multi": False, "rerank": False, "parent": False, "compress": False},
        {"name": "hybrid_reranking", "mode": "hybrid", "rewrite": False, "multi": False, "rerank": True, "parent": False, "compress": False},
        {"name": "hybrid_query_rewrite", "mode": "hybrid", "rewrite": True, "multi": False, "rerank": True, "parent": False, "compress": False},
        {"name": "hybrid_multi_query", "mode": "hybrid", "rewrite": True, "multi": True, "rerank": True, "parent": False, "compress": False},
        {"name": "advanced_full", "mode": "hybrid", "rewrite": True, "multi": True, "rerank": True, "parent": True, "compress": True},
    ]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    try:
        with factory() as db:
            for config in configs:
                retrieval = RetrievalService(
                    embed_query=embedder.embed_query,
                    embedding_dimension=settings.EMBEDDING_DIMENSION,
                    reranker=CrossEncoderReranker(model_name=settings.RERANKER_MODEL) if config["rerank"] else None,
                    query_rewriter=QueryRewriter(generate=lambda prompt: model.complete(
                        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=160).text),
                    multi_query_generator=MultiQueryGenerator(generate=lambda prompt: model.complete(
                        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=220).text,
                        max_queries=settings.MULTI_QUERY_MAX_QUERIES),
                )
                def evaluate(case):
                    request_started = perf_counter()
                    result = retrieval.search(
                        db, user_id=UUID(args.user_id), workspace_id=UUID(case.workspace_id or args.workspace_id),
                        query=case.query,
                        document_ids=[UUID(x) for x in (case.document_ids or ())] or None,
                        candidate_k=args.candidate_k, final_k=args.final_k,
                        retrieval_mode=config["mode"], use_query_rewrite=config["rewrite"],
                        use_multi_query=config["multi"], use_reranker=config["rerank"],
                        use_parent_expansion=config["parent"], use_compression=config["compress"],
                    )
                    out = {
                        "retrieved_chunk_ids": [str(item.chunk_id) for item in result.chunks],
                        "retrieved_source_names": [str(item.metadata.get("filename") or "") for item in result.chunks],
                        "abstained": not result.evidence_sufficient,
                    }
                    if args.generate and result.chunks and result.evidence_sufficient:
                        generated = model.complete(build_rag_messages(query=case.query, retrieval=result, history=[]),
                                                    temperature=0.0, max_tokens=1000)
                        answer, citations = validate_citations(generated.text, result)
                        out.update({"answer": answer, "prompt_tokens": generated.prompt_tokens,
                                    "completion_tokens": generated.completion_tokens})
                        expected = set(case.relevant_source_names)
                        cited = {c.get("filename") for c in citations if c.get("source_type") == "document"}
                        if expected:
                            out["citation_precision"] = len(cited & expected) / max(len(cited), 1)
                            out["citation_coverage"] = len(cited & expected) / len(expected)
                        elif case.relevant_chunk_ids:
                            cited_ids = {c.get("chunk_id") for c in citations if c.get("source_type") == "document"}
                            relevant_ids = set(case.relevant_chunk_ids)
                            out["citation_precision"] = len(cited_ids & relevant_ids) / max(len(cited_ids), 1)
                            out["citation_coverage"] = len(cited_ids & relevant_ids) / len(relevant_ids)
                        evidence_text = "\n\n".join(item.context_content for item in result.chunks)
                        faithfulness = AnswerVerifier(model).verify(question=case.query, answer=answer, evidence_text=evidence_text)
                        if faithfulness.get("verified") is not None:
                            out["faithfulness"] = float(faithfulness["verified"])
                    out["latency_ms"] = (perf_counter() - request_started) * 1000
                    return out
                report = run_evaluation(cases, evaluate, k=args.final_k, configuration=config["name"])
                write_report(report, output_dir / f"{config['name']}.json")
                write_markdown_report(report, output_dir / f"{config['name']}.md")
                comparison_rows.append({"configuration": config["name"], **report["summary"]})
                print(json.dumps({"configuration": config["name"], "case_count": report["case_count"],
                                  "summary": report["summary"]}, indent=2))
        comparison = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": args.workspace_id,
            "case_count": len(cases),
            "configurations": comparison_rows,
        }
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        columns = ["configuration", "recall_at_k", "precision_at_k", "mrr", "ndcg",
                   "citation_precision", "citation_coverage", "faithfulness",
                   "latency_p50_ms", "latency_p95_ms", "mean_total_tokens"]
        lines = ["# RAGNexus Retrieval Configuration Comparison", "",
                 f"Generated: {comparison['generated_at']}", f"Workspace: {args.workspace_id}",
                 f"Evaluation cases: {len(cases)}", "", "| " + " | ".join(columns) + " |",
                 "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in comparison_rows:
            lines.append("| " + " | ".join(
                f"{row.get(column, ''):.4f}" if isinstance(row.get(column), (float, int))
                else str(row.get(column, "")) for column in columns
            ) + " |")
        lines += ["", "Each configuration also has a per-case JSON and Markdown report in this directory.",
                  "Answer, citation and faithfulness columns are populated only when `--generate` is used."]
        (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
