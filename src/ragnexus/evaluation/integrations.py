"""Optional Ragas and DeepEval adapters.

The core evaluation runner remains dependency-light and deterministic. These
adapters let teams pass their configured judge metrics without forcing either
framework into the production API process.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Sequence


def optional_framework_status() -> dict[str, bool]:
    return {
        "ragas": importlib.util.find_spec("ragas") is not None,
        "deepeval": importlib.util.find_spec("deepeval") is not None,
        "phoenix": importlib.util.find_spec("phoenix") is not None,
    }


def build_ragas_samples(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map RAGNexus evaluation records to Ragas' standard sample fields."""
    samples = []
    for item in records:
        samples.append({
            "user_input": item.get("query") or item.get("user_input", ""),
            "response": item.get("answer") or item.get("response", ""),
            "retrieved_contexts": item.get("context_texts") or item.get("retrieved_contexts") or [],
            "reference": item.get("reference_answer") or item.get("reference"),
        })
    return samples



def default_ragas_metrics() -> list[Any]:
    """Construct commonly used Ragas metrics from the installed Ragas version."""
    try:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
            NoiseSensitivity,
            ContextEntityRecall,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install a Ragas version exposing the standard metric classes from requirements-evaluation.txt."
        ) from exc
    return [
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        ResponseRelevancy(),
        NoiseSensitivity(),
        ContextEntityRecall(),
    ]


def run_ragas(
    records: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[Any],
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, Any]:
    """Run caller-selected Ragas metrics; inject the selected-provider judge."""
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Ragas is optional. Install it with `pip install -r requirements-evaluation.txt`."
        ) from exc

    dataset = EvaluationDataset.from_list(build_ragas_samples(records))
    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": list(metrics)}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    result = evaluate(**kwargs)
    if hasattr(result, "to_pandas"):
        return {"framework": "ragas", "results": result.to_pandas().to_dict(orient="records")}
    return {"framework": "ragas", "results": result}


def build_deepeval_test_cases(records: Sequence[dict[str, Any]]) -> list[Any]:
    """Convert RAGNexus rows to DeepEval LLMTestCase objects."""
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError(
            "DeepEval is optional. Install it with `pip install -r requirements-evaluation.txt`."
        ) from exc

    cases = []
    for item in records:
        cases.append(LLMTestCase(
            input=item.get("query") or item.get("input", ""),
            actual_output=item.get("answer") or item.get("actual_output", ""),
            expected_output=item.get("reference_answer") or item.get("expected_output", ""),
            retrieval_context=item.get("context_texts") or item.get("retrieval_context") or [],
        ))
    return cases



def default_deepeval_metrics(*, model: Any | None = None) -> list[Any]:
    """Construct common DeepEval RAG metrics; pass a provider-backed judge model."""
    try:
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install DeepEval with `pip install -r requirements-evaluation.txt`."
        ) from exc
    kwargs = {"model": model} if model is not None else {}
    return [
        FaithfulnessMetric(**kwargs),
        AnswerRelevancyMetric(**kwargs),
        ContextualPrecisionMetric(**kwargs),
        ContextualRecallMetric(**kwargs),
    ]


def run_deepeval(test_cases: Sequence[Any], *, metrics: Sequence[Any]) -> dict[str, Any]:
    """Run caller-selected DeepEval metrics and return its native result."""
    try:
        from deepeval import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "DeepEval is optional. Install it with `pip install -r requirements-evaluation.txt`."
        ) from exc
    result = evaluate(test_cases=list(test_cases), metrics=list(metrics))
    return {"framework": "deepeval", "results": result}
