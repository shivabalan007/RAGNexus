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
    source_document: str | None = None
    source_documents: tuple[str, ...] = ()
    expected_source_ids: tuple[str, ...] = ()
    evidence_quote: str | None = None
    evidence_quotes: dict[str, str | list[str]] | None = None
    question_type: str | None = None
    mapping_status: str | None = None
    prior_turns: tuple[dict, ...] = ()


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
                    source_document=raw.get("source_document"),
                    source_documents=tuple(str(x) for x in raw.get("source_documents", [])),
                    expected_source_ids=tuple(str(x) for x in raw.get("expected_source_ids", [])),
                    evidence_quote=raw.get("evidence_quote"),
                    evidence_quotes=raw.get("evidence_quotes"),
                    question_type=raw.get("question_type"),
                    mapping_status=raw.get("mapping_status"),
                    prior_turns=tuple(raw.get("prior_turns", [])),
                )
                if not case.case_id or not case.query.strip():
                    raise ValueError("case_id and non-empty query are required")
                # Real chunk UUIDs are assigned by the database during ingestion.
                # Source-grounded benchmark records may be loaded before mapping.
                if not case.should_abstain and not case.relevant_chunk_ids and not (case.source_document or case.source_documents):
                    raise ValueError(
                        "answerable cases require relevant_chunk_ids or a source_document awaiting chunk mapping"
                    )
                if case.should_abstain and case.relevant_chunk_ids:
                    raise ValueError("unanswerable cases must not declare relevant_chunk_ids")
                cases.append(case)
            except Exception as exc:
                raise ValueError(f"Invalid evaluation dataset at line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique")
    return cases
