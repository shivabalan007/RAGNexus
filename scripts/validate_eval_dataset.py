from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragnexus.evaluation.dataset import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="evals/datasets/ragnexus_ground_truth_100.jsonl")
args = parser.parse_args()
cases = load_jsonl(ROOT / args.dataset)
print(f"Validated {len(cases)} evaluation cases.")
answerable = [case for case in cases if not case.should_abstain]
mapped = [case for case in answerable if case.relevant_chunk_ids]
print(f"Answerable cases: {len(answerable)}")
print(f"Mapped answerable cases: {len(mapped)} ({len(mapped) / max(len(answerable), 1):.1%})")
print(f"Unanswerable cases: {len(cases) - len(answerable)}")
if not 100 <= len(cases) <= 200:
    print("WARNING: recommended benchmark size is 100–200 cases.")
