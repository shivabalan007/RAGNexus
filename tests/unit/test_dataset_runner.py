import json
from pathlib import Path

import pytest
from ragnexus.evaluation.dataset import load_jsonl
from ragnexus.evaluation.runner import run_evaluation
from ragnexus.evaluation.thresholds import check_thresholds


def test_dataset_and_runner(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"case_id":"c1", "query":"q", "relevant_chunk_ids":["x"]}) + "\n", encoding="utf-8")
    cases = load_jsonl(path)
    report = run_evaluation(cases, lambda case: {"retrieved_chunk_ids":["x"]})
    assert report["case_count"] == 1
    assert report["summary"]["recall_at_k"] == 1.0
    assert check_thresholds(report, {"mrr": 0.9}) == []


def test_duplicate_case_ids_rejected(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    item = {"case_id":"same", "query":"q", "relevant_chunk_ids":["x"]}
    path.write_text(json.dumps(item) + "\n" + json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_jsonl(path)
