from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class PipelineTrace:
    """Request-scoped, non-sensitive timing/usage record. Do not store raw prompts."""
    request_id: str
    workspace_id: str
    session_id: str
    stages_ms: dict[str, float] = field(default_factory=dict)
    retrieved_count: int = 0
    abstained: bool = False
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageTimer:
    def __init__(self, trace: PipelineTrace, stage: str):
        self.trace, self.stage = trace, stage
        self.started = 0.0

    def __enter__(self):
        self.started = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trace.stages_ms[self.stage] = round((perf_counter() - self.started) * 1000, 3)
        return False
