from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Sequence
import json

from ragnexus.evaluation.dataset import EvaluationCase
from ragnexus.evaluation.metrics import exact_match, expected_fact_coverage, retrieval_metrics, token_f1

EvaluateQuery = Callable[[EvaluationCase], dict]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def run_evaluation(cases: Sequence[EvaluationCase], evaluate_query: EvaluateQuery, *, k: int = 5,
                   configuration: str | None = None) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    rows: list[dict] = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    latency_values: list[float] = []
    token_values: list[float] = []
    for case in cases:
        output = evaluate_query(case)
        retrieved_ids = [str(x) for x in output.get("retrieved_chunk_ids", [])]
        retrieved_names = [str(x) for x in output.get("retrieved_source_names", [])]
        actual_abstain = bool(output.get("abstained", False))
        row = {"case_id": case.case_id, "query": case.query, "retrieved_chunk_ids": retrieved_ids,
               "retrieved_source_names": retrieved_names, "expected_abstain": case.should_abstain,
               "actual_abstain": actual_abstain}
        answerability_accuracy = float(actual_abstain == case.should_abstain)
        row["answerability_accuracy"] = answerability_accuracy
        metric_values["answerability_accuracy"].append(answerability_accuracy)
        if case.should_abstain:
            score = float(actual_abstain)
            row["abstention_accuracy"] = score
            metric_values["abstention_accuracy"].append(score)
        else:
            relevant_ids = list(case.relevant_chunk_ids)
            if relevant_ids:
                metrics = retrieval_metrics(relevant_ids, retrieved_ids, k)
            else:
                metrics = retrieval_metrics(case.relevant_source_names, retrieved_names, k)
            row.update(metrics)
            for name, value in metrics.items():
                metric_values[name].append(value)
        if case.reference_answer is not None and output.get("answer") is not None:
            em = exact_match(str(output["answer"]), case.reference_answer)
            f1 = token_f1(str(output["answer"]), case.reference_answer)
            row.update({"answer_exact_match": em, "answer_token_f1": f1})
            metric_values["answer_exact_match"].append(em)
            metric_values["answer_token_f1"].append(f1)
        if case.expected_facts and output.get("answer") is not None:
            coverage = expected_fact_coverage(str(output["answer"]), case.expected_facts)
            row["expected_fact_coverage_lexical"] = coverage
            metric_values["expected_fact_coverage_lexical"].append(coverage)
        if output.get("citation_precision") is not None:
            value = float(output["citation_precision"])
            row["citation_precision"] = value
            metric_values["citation_precision"].append(value)
        if output.get("citation_coverage") is not None:
            value = float(output["citation_coverage"])
            row["citation_coverage"] = value
            metric_values["citation_coverage"].append(value)
        if output.get("faithfulness") is not None:
            value = float(output["faithfulness"])
            row["faithfulness"] = value
            metric_values["faithfulness"].append(value)
        if output.get("latency_ms") is not None:
            value = float(output["latency_ms"])
            row["latency_ms"] = value
            latency_values.append(value)
        total_tokens = (output.get("prompt_tokens") or 0) + (output.get("completion_tokens") or 0)
        if total_tokens:
            row["total_tokens"] = total_tokens
            token_values.append(float(total_tokens))
        rows.append(row)
    summary = {name: sum(values) / len(values) for name, values in metric_values.items() if values}
    if latency_values:
        summary.update({"latency_p50_ms": median(latency_values), "latency_p95_ms": _percentile(latency_values, 95),
                        "latency_mean_ms": sum(latency_values) / len(latency_values)})
    if token_values:
        summary["mean_total_tokens"] = sum(token_values) / len(token_values)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
            "case_count": len(cases), "k": k, "summary": summary, "cases": rows}


def write_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_report(report: dict, output_path: str | Path) -> None:
    summary = report.get("summary", {})
    lines = [f"# RAGNexus Evaluation Report", "", f"Generated: {report.get('generated_at')}",
             f"Configuration: {report.get('configuration') or 'unspecified'}",
             f"Test cases: {report.get('case_count', 0)}", "", "## Summary", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.4f} |" if isinstance(value, (float, int)) else f"| {key} | {value} |"
                 for key, value in summary.items())
    lines += ["", "## Per-case results", "", "| Case | Query | Recall@K | MRR | Latency ms | Abstained |",
              "|---|---|---:|---:|---:|---|"]
    for row in report.get("cases", []):
        lines.append(f"| {row.get('case_id')} | {str(row.get('query','')).replace('|','\\|')} | "
                     f"{row.get('recall_at_k','')} | {row.get('mrr','')} | {row.get('latency_ms','')} | {row.get('actual_abstain')} |")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
