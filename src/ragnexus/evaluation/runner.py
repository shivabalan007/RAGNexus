from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ragnexus.evaluation.dataset import EvaluationCase
from ragnexus.evaluation.metrics import exact_match, retrieval_metrics, token_f1


# Adapter contract: callback returns ranked chunk IDs and optional answer/abstention.
EvaluateQuery = Callable[[EvaluationCase], dict]


def run_evaluation(cases: Sequence[EvaluationCase], evaluate_query: EvaluateQuery, *, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    rows: list[dict] = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        output = evaluate_query(case)
        retrieved_ids = [str(x) for x in output.get("retrieved_chunk_ids", [])]
        actual_abstain = bool(output.get("abstained", False))
        row = {"case_id": case.case_id, "query": case.query, "retrieved_chunk_ids": retrieved_ids,
               "expected_abstain": case.should_abstain, "actual_abstain": actual_abstain}
        if case.should_abstain:
            abstain_correct = float(actual_abstain)
            row["abstention_accuracy"] = abstain_correct
            metric_values["abstention_accuracy"].append(abstain_correct)
        else:
            metrics = retrieval_metrics(case.relevant_chunk_ids, retrieved_ids, k)
            row.update(metrics)
            for name, value in metrics.items():
                metric_values[name].append(value)
        if case.reference_answer is not None and output.get("answer") is not None:
            em = exact_match(str(output["answer"]), case.reference_answer)
            f1 = token_f1(str(output["answer"]), case.reference_answer)
            row.update({"answer_exact_match": em, "answer_token_f1": f1})
            metric_values["answer_exact_match"].append(em)
            metric_values["answer_token_f1"].append(f1)
        rows.append(row)
    summary = {name: sum(values) / len(values) for name, values in metric_values.items() if values}
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases), "k": k,
            "summary": summary, "cases": rows}


def write_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
