from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.runner import run_evaluation, write_report


def main():
    parser = argparse.ArgumentParser(description="Run the 100-question benchmark against a live RAGNexus API.")
    parser.add_argument("--dataset", default="evals/datasets/ragnexus_ground_truth_100.jsonl")
    parser.add_argument("--base-url", default=os.getenv("RAGNEXUS_API_URL", "http://localhost:8000/api/v1"))
    parser.add_argument("--token", default=os.getenv("RAGNEXUS_API_TOKEN"))
    parser.add_argument("--workspace-id", default=os.getenv("RAGNEXUS_WORKSPACE_ID"))
    parser.add_argument("--output", default="evals/reports/ragnexus-evaluation.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pipeline-mode", choices=("standard", "agentic"), default="standard")
    parser.add_argument("--enable-web-search", action="store_true", help="Allow Agentic RAG to use configured Tavily search")
    parser.add_argument("--llm-judge", action="store_true", help="Score semantic answer/context metrics using the configured LLM provider (additional cost/latency).")
    args = parser.parse_args()
    if not args.token:
        parser.error("Set RAGNEXUS_API_TOKEN or pass --token")
    cases = load_jsonl(args.dataset)
    if not args.workspace_id and not any(case.workspace_id for case in cases):
        parser.error("Set RAGNEXUS_WORKSPACE_ID or map workspace IDs into the dataset")

    judge = None
    if args.llm_judge:
        from ragnexus.config import get_settings
        from ragnexus.llm.providers.factory import build_chat_model
        from ragnexus.evaluation.judge import ProviderMetricJudge
        judge = ProviderMetricJudge(build_chat_model(get_settings()))

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {args.token}", "Accept": "application/json"})

    def evaluate_case(case):
        workspace_id = case.workspace_id or args.workspace_id
        if not workspace_id:
            return {"error": "workspace_id_missing"}

        session_id = None
        prior_retrieved_ids = set()
        try:
            scope = "selected" if case.document_ids else "workspace"
            created = session.post(
                f"{args.base_url.rstrip('/')}/chat/sessions",
                json={
                    "workspace_id": workspace_id,
                    "visibility": "private",
                    "document_scope": scope,
                    "document_ids": list(case.document_ids),
                    "title": f"Evaluation {case.case_id}",
                },
                timeout=30,
            )
            created.raise_for_status()
            session_id = created.json()["id"]

            for prior in case.prior_turns:
                prior_query = prior.get("content") if isinstance(prior, dict) else str(prior)
                if prior_query:
                    prior_response = session.post(
                        f"{args.base_url.rstrip('/')}/chat/messages",
                        json={"session_id": session_id, "query": prior_query, "candidate_k": 20, "final_k": args.k,
                              "pipeline_mode": args.pipeline_mode, "enable_web_search": args.enable_web_search},
                        timeout=180,
                    )
                    prior_response.raise_for_status()
                    prior_payload = prior_response.json()
                    prior_diagnostics = prior_payload.get("diagnostics") or {}
                    prior_retrieved_ids.update(
                        str(item.get("chunk_id"))
                        for item in prior_diagnostics.get("retrieved_chunks", [])
                        if item.get("chunk_id")
                    )

            response = session.post(
                f"{args.base_url.rstrip('/')}/chat/messages",
                json={"session_id": session_id, "query": case.query, "candidate_k": 20, "final_k": args.k,
                      "pipeline_mode": args.pipeline_mode, "enable_web_search": args.enable_web_search},
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            diagnostics = result.get("diagnostics") or {}
            retrieved = diagnostics.get("retrieved_chunks") or []
            cited = result.get("citations") or []
            output = {
                "answer": result.get("answer", ""),
                "abstained": bool(result.get("abstained")),
                "retrieved_chunk_ids": [item.get("chunk_id") for item in retrieved if item.get("chunk_id")],
                "cited_source_ids": [item.get("filename") or item.get("document_id") for item in cited],
                "valid_source_ids": [item.get("filename") or item.get("document_id") for item in retrieved],
                "context_texts": [item.get("preview", "") for item in retrieved if item.get("preview")],
                "latency_ms": result.get("latency_ms"),
                "retrieval_latency_ms": diagnostics.get("retrieval_latency_ms"),
                "generation_latency_ms": diagnostics.get("generation_latency_ms"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "retry_count": result.get("retry_count", 0),
            }
            if case.prior_turns:
                current_ids = set(output["retrieved_chunk_ids"])
                union_ids = current_ids | prior_retrieved_ids
                output["retrieval_consistency"] = (
                    len(current_ids & prior_retrieved_ids) / len(union_ids)
                    if union_ids else 0.0
                )
            if judge is not None:
                try:
                    output["judge_metrics"] = judge.score(
                        query=case.query,
                        answer=result.get("answer", ""),
                        contexts=output["context_texts"],
                        reference_answer=case.reference_answer,
                    )
                    if case.prior_turns:
                        output["followup_context_relevance"] = output["judge_metrics"].get("answer_relevancy")
                        output["multi_turn_faithfulness"] = output["judge_metrics"].get("faithfulness")
                except Exception as judge_exc:
                    output["judge_error"] = f"{type(judge_exc).__name__}: {judge_exc}"
            return output
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "abstained": False}
        finally:
            if session_id:
                # Evaluation sessions are disposable; do not clutter user chat history.
                try:
                    session.delete(f"{args.base_url.rstrip('/')}/chat/sessions/{session_id}", timeout=30)
                except requests.RequestException:
                    pass

    report = run_evaluation(cases, evaluate_case, k=args.k)
    report["pipeline_mode"] = args.pipeline_mode
    report["web_search_enabled"] = bool(args.enable_web_search)
    write_report(report, args.output)
    print(f"Pipeline: {args.pipeline_mode}")
    print(f"Cases: {report['case_count']}")
    print(f"Answerable: {report['answerable_case_count']}")
    print(f"Unanswerable: {report['unanswerable_case_count']}")
    print(f"Chunk mapping coverage: {report['summary'].get('retrieval_mapping_coverage', 0):.1%}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
