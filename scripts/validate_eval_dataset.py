from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragnexus.evaluation.dataset import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="evals/datasets/ground_truth.jsonl")
args = parser.parse_args()
cases = load_jsonl(ROOT / args.dataset)
print(f"Validated {len(cases)} labeled evaluation cases.")
if not 100 <= len(cases) <= 200:
    print("WARNING: recommended benchmark size is 100–200 cases.")
