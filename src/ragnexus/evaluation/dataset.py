from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    reference_answer: str | None = None
    workspace_id: str | None = None
    document_ids: tuple[str, ...] = ()
    should_abstain: bool = False


def load_jsonl(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                case = EvaluationCase(
                    case_id=str(raw["case_id"]),
                    query=str(raw["query"]),
                    relevant_chunk_ids=tuple(str(x) for x in raw.get("relevant_chunk_ids", [])),
                    reference_answer=raw.get("reference_answer"),
                    workspace_id=raw.get("workspace_id"),
                    document_ids=tuple(str(x) for x in raw.get("document_ids", [])),
                    should_abstain=bool(raw.get("should_abstain", False)),
                )
                if not case.case_id or not case.query.strip():
                    raise ValueError("case_id and non-empty query are required")
                if not case.should_abstain and not case.relevant_chunk_ids:
                    raise ValueError("non-abstention cases require relevant_chunk_ids")
                cases.append(case)
            except Exception as exc:
                raise ValueError(f"Invalid evaluation dataset at line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique")
    return cases
