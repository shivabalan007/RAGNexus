# RAGNexus evaluation datasets

`ground_truth_120.template.jsonl` is a 120-case **labeling template**, not a completed ground-truth benchmark. It contains draft question patterns and is ignored by the evaluation loader until each entry has `status` changed from `draft` to `ready` and has been reviewed by a human.

For each ready case, fill:

- `query`: a real question grounded in your corpus.
- `reference_answer`: the expected answer or a concise fact list.
- `relevant_chunk_ids`: stable chunk UUIDs from the current indexed corpus, or `relevant_source_names` when evaluating at document level.
- `expected_facts`: key facts a correct answer must include.
- `should_abstain`: set true only when the allowed corpus does not support an answer.
- `document_ids`: optional document allow-list for selected-document tests.

Chunk UUIDs change when a document is re-ingested or re-versioned. Re-label the dataset after such changes, or prefer `relevant_source_names` for a less precise document-level benchmark. Do not use private workspace data in a public portfolio dataset without permission.

Run `python scripts/validate_eval_dataset.py --dataset evals/datasets/ground_truth.jsonl` after creating the ready dataset.
