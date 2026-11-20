from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk


def normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def map_dataset(dataset_path: Path, workspace_id: str) -> tuple[int, int]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    workspace_uuid = UUID(workspace_id)
    engine = create_database_engine(get_settings())
    factory = create_session_factory(engine)
    mapped = 0
    unresolved = 0
    try:
        with factory() as db:
            docs = db.scalars(
                select(Document).where(
                    Document.workspace_id == workspace_uuid,
                    Document.status == "indexed",
                    Document.document_metadata["is_current"].as_boolean().is_(True),
                )
            ).all()
            by_name: dict[str, list[Document]] = {}
            for doc in docs:
                by_name.setdefault(doc.filename.casefold(), []).append(doc)

            for row in rows:
                if row.get("should_abstain"):
                    row["workspace_id"] = workspace_id
                    row["document_ids"] = []
                    row["relevant_chunk_ids"] = []
                    row["mapping_status"] = "not_applicable"
                    continue

                row["document_ids"] = []
                row["relevant_chunk_ids"] = []
                source_names = row.get("source_documents") or (
                    [row.get("source_document")] if row.get("source_document") else []
                )
                evidence_map = row.get("evidence_quotes") or {
                    source_names[0]: row.get("evidence_quote") or ""
                }
                mapped_ids = []
                mapped_doc_ids = []
                unresolved_sources = []

                for source_name in source_names:
                    candidates = by_name.get(str(source_name).casefold(), [])
                    if len(candidates) != 1:
                        unresolved_sources.append(
                            "unresolved_document" if not candidates else "ambiguous_document"
                        )
                        continue
                    document = candidates[0]
                    raw_quotes = evidence_map.get(source_name, "")
                    quotes = raw_quotes if isinstance(raw_quotes, list) else [raw_quotes]
                    normalized_quotes = [normalize(quote) for quote in quotes if quote]
                    chunks = db.scalars(
                        select(DocumentChunk).where(DocumentChunk.document_id == document.id)
                    ).all()
                    source_matches = []
                    for quote in normalized_quotes:
                        quote_matches = [
                            chunk for chunk in chunks
                            if quote in normalize(chunk.content)
                        ]
                        if not quote_matches:
                            unresolved_sources.append("unresolved_evidence")
                        source_matches.extend(quote_matches)
                    if not source_matches:
                        continue
                    mapped_ids.extend(str(chunk.id) for chunk in source_matches)
                    mapped_doc_ids.append(str(document.id))

                row["workspace_id"] = workspace_id
                if unresolved_sources:
                    # Partial evidence mappings are not valid ground truth for
                    # multi-source/multi-span cases; keep the row unresolved.
                    row["document_ids"] = []
                    row["relevant_chunk_ids"] = []
                    row["mapping_status"] = ",".join(sorted(set(unresolved_sources)))
                    unresolved += 1
                elif mapped_ids:
                    row["document_ids"] = sorted(set(mapped_doc_ids))
                    row["relevant_chunk_ids"] = sorted(set(mapped_ids))
                    row["mapping_status"] = "mapped"
                    mapped += 1
                else:
                    row["document_ids"] = []
                    row["relevant_chunk_ids"] = []
                    row["mapping_status"] = "unresolved_evidence"
                    unresolved += 1

        dataset_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return mapped, unresolved
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Map benchmark evidence to actual active RAGNexus chunk UUIDs.")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument(
        "--dataset",
        default="evals/datasets/ragnexus_ground_truth_100.jsonl",
    )
    args = parser.parse_args()
    mapped, unresolved = map_dataset(Path(args.dataset), args.workspace_id)
    print(f"Mapped answerable cases: {mapped}")
    print(f"Unresolved cases: {unresolved}")
    print("No chunk IDs are generated by this script; only exact evidence-to-chunk matches are written.")


if __name__ == "__main__":
    main()
