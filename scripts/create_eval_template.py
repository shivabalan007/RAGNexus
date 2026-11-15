from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = [
    ("direct_fact", "What is [ENTITY] used for?"),
    ("direct_fact", "What is the purpose of [PROJECT OR SYSTEM]?"),
    ("direct_fact", "Which component is responsible for [FUNCTION]?"),
    ("direct_fact", "What value or threshold is configured for [SETTING]?"),
    ("direct_fact", "Who is the escalation contact for [PROCESS]?"),
    ("direct_fact", "When was [EVENT OR VERSION] introduced?"),
    ("direct_fact", "Where is [COMPONENT] located?"),
    ("direct_fact", "Which model, provider, or library is used for [TASK]?"),
    ("direct_fact", "What are the supported formats for [FEATURE]?"),
    ("direct_fact", "How many [COMPONENTS] are listed in [DOCUMENT]?"),
    ("contextual_followup", "What is its primary responsibility?"),
    ("contextual_followup", "Which version contains that change?"),
    ("contextual_followup", "What happens when that threshold is exceeded?"),
    ("contextual_followup", "Who handles that escalation?"),
    ("contextual_followup", "Can you explain the second component?"),
    ("multi_hop", "How does [COMPONENT A] interact with [COMPONENT B]?"),
    ("multi_hop", "Which configuration controls [BEHAVIOUR], and what is its value?"),
    ("multi_hop", "Compare the responsibilities of [COMPONENT A] and [COMPONENT B]."),
    ("multi_hop", "What sequence of steps leads from [INPUT] to [OUTPUT]?"),
    ("multi_hop", "Which two documents describe [TOPIC], and how do they differ?"),
    ("conflict", "Two sources give different values for [FACT]. What does each source say?"),
    ("conflict", "Which current document version contains [UPDATED FACT]?"),
    ("abstention", "What is [UNSUPPORTED FACT] if none of the uploaded files mention it?"),
    ("abstention", "Which private information is stored outside my selected documents?"),
    ("scope", "Answer this using only [SELECTED DOCUMENT]."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a 120-case ground-truth labeling template.")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--output", default="evals/datasets/ground_truth_120.template.jsonl")
    args = parser.parse_args()
    if not 100 <= args.count <= 200:
        raise SystemExit("count must be between 100 and 200")
    rows = []
    for index in range(args.count):
        category, prompt = TEMPLATES[index % len(TEMPLATES)]
        rows.append({
            "case_id": f"manual-{index + 1:03d}",
            "status": "draft",
            "query": prompt,
            "question_type": category,
            "reference_answer": None,
            "expected_facts": [],
            "relevant_chunk_ids": [],
            "relevant_source_names": [],
            "document_ids": [],
            "should_abstain": category == "abstention",
            "labeling_note": "Replace placeholders with a real question; add reference_answer and at least one relevant_chunk_id or relevant_source_name. Review expected abstention manually.",
        })
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Created {len(rows)} draft cases at {path}. Draft cases are ignored by the benchmark until labeled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
