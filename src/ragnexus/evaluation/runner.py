from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ragnexus.evaluation.dataset import EvaluationCase
from ragnexus.evaluation.metrics import (
    answerability_metrics, citation_metrics, context_quality_metrics,
    exact_match, percentile, retrieval_metrics, token_f1,
)

# Adapter contract: return actual ranked chunk IDs, answer/citations and any
# optional Ragas/DeepEval judge scores or agentic/conversation observations.
EvaluateQuery = Callable[[EvaluationCase], dict]


def run_evaluation(
    cases: Sequence[EvaluationCase],
    evaluate_query: EvaluateQuery,
    *,
    k: int = 5,
) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")

    rows: list[dict] = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)

    for case in cases:
        output = evaluate_query(case) or {}
        retrieved_ids = [str(x) for x in output.get("retrieved_chunk_ids", [])]
        actual_abstain = bool(output.get("abstained", False))
        answer = output.get("answer")
        row = {
            "case_id": case.case_id,
            "query": case.query,
            "source_document": case.source_document,
            "source_documents": list(case.source_documents),
            "question_type": case.question_type,
            "mapping_status": case.mapping_status,
            "prior_turn_count": len(case.prior_turns),
            "retrieved_chunk_ids": retrieved_ids,
            "expected_abstain": case.should_abstain,
            "actual_abstain": actual_abstain,
            "error": output.get("error"),
        }

        # Do not score retrieval as zero when this benchmark row has not yet
        # been mapped to real database chunk UUIDs.
        if case.relevant_chunk_ids:
            retrieval = retrieval_metrics(case.relevant_chunk_ids, retrieved_ids, k)
            row.update(retrieval)
            for name, value in retrieval.items():
                metric_values[name].append(value)
        else:
            row["retrieval_metrics_status"] = (
                "not_applicable" if case.should_abstain else "pending_real_chunk_mapping"
            )

        answer_correct = None
        if case.reference_answer is not None and answer is not None:
            em = exact_match(str(answer), case.reference_answer)
            f1 = token_f1(str(answer), case.reference_answer)
            answer_correct = bool(em or f1 >= 0.5)
            row.update({"answer_exact_match": em, "answer_token_f1": f1})
            metric_values["answer_exact_match"].append(em)
            metric_values["answer_token_f1"].append(f1)

        answerability = answerability_metrics(
            expected_abstain=case.should_abstain,
            actual_abstain=actual_abstain,
            answer_correct=answer_correct,
        )
        row.update(answerability)
        for name, value in answerability.items():
            # Answerable accuracy is only meaningful for answerable cases.
            if name != "answerable_question_accuracy" or not case.should_abstain:
                metric_values[name].append(value)

        cited_ids = output.get("cited_source_ids", [])
        if case.expected_source_ids or cited_ids:
            citation = citation_metrics(
                case.expected_source_ids,
                cited_ids,
                output.get("valid_source_ids", cited_ids),
            )
            row.update(citation)
            for name, value in citation.items():
                metric_values[name].append(value)

        context_texts = output.get("context_texts") or []
        if context_texts:
            context = context_quality_metrics(
                case.query,
                context_texts,
                case.evidence_quote or " ".join(
                    quote
                    for value in (case.evidence_quotes or {}).values()
                    for quote in (value if isinstance(value, list) else [value])
                ),
            )
            row.update(context)
            for name, value in context.items():
                metric_values[name].append(value)

        # Conversational and agentic assertions are supplied by the live
        # orchestration adapter, not fabricated by an LLM judge.
        for name in (
            "followup_context_relevance", "multi_turn_faithfulness",
            "retrieval_consistency", "routing_accuracy", "tool_call_accuracy",
            "task_completion", "corrective_retrieval_success",
        ):
            if name in output:
                raw_value = output[name]
                value = float(raw_value) if isinstance(raw_value, (int, float)) else float(bool(raw_value))
                row[name] = value
                metric_values[name].append(value)

        judge_metrics = output.get("judge_metrics") or {}
        for name, value in judge_metrics.items():
            if name == "judge_latency_ms" and isinstance(value, (int, float)):
                latencies[name] .append(float(value))
            elif isinstance(value, (int, float)):
                row[name] = float(value)
                metric_values[name].append(float(value))
            elif isinstance(value, str) and name == "judge_model":
                row[name] = value

        for name in (
            "latency_ms", "retrieval_latency_ms", "generation_latency_ms",
            "dense_latency_ms", "bm25_latency_ms", "rerank_latency_ms",
        ):
            if output.get(name) is not None:
                latencies[name].append(float(output[name]))
                row[name] = float(output[name])

        for name in ("prompt_tokens", "completion_tokens", "retry_count"):
            if output.get(name) is not None:
                row[name] = float(output[name])
                metric_values[name].append(float(output[name]))

        rows.append(row)

    summary = {
        name: sum(values) / len(values)
        for name, values in metric_values.items()
        if values
    }
    for name, values in latencies.items():
        summary[f"{name}_p50"] = percentile(values, 50)
        summary[f"{name}_p95"] = percentile(values, 95)
        summary[f"{name}_mean"] = sum(values) / len(values)

    mapped = sum(bool(case.relevant_chunk_ids) for case in cases if not case.should_abstain)
    answerable = sum(not case.should_abstain for case in cases)
    summary["retrieval_mapping_coverage"] = mapped / max(answerable, 1)
    summary["error_rate"] = sum(bool(row.get("error")) for row in rows) / max(len(rows), 1)
    summary["retry_rate"] = sum(float(row.get("retry_count", 0)) > 0 for row in rows) / max(len(rows), 1)
    summary["prompt_tokens_total"] = sum(float(row.get("prompt_tokens", 0)) for row in rows)
    summary["completion_tokens_total"] = sum(float(row.get("completion_tokens", 0)) for row in rows)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "answerable_case_count": answerable,
        "unanswerable_case_count": len(cases) - answerable,
        "retrieval_mapped_case_count": mapped,
        "k": k,
        "summary": summary,
        "cases": rows,
    }


def write_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
