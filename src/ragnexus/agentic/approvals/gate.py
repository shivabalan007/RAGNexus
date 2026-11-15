from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ApprovalRequired:
    action_name: str
    arguments: dict[str, Any]
    rationale: str


class HumanApprovalRequired(RuntimeError):
    def __init__(self, request: ApprovalRequired):
        super().__init__(f"Human approval is required before executing '{request.action_name}'.")
        self.request = request


def enforce_tool_approval(
    action_name: str,
    arguments: dict[str, Any],
    *,
    sensitive: bool,
    approval_status: str | None = None,
    rationale: str = "This action can modify data or affect an external system.",
) -> None:
    """Call this immediately before any mutating/privileged agent tool executes."""
    if not sensitive:
        return
    if approval_status != "approved":
        raise HumanApprovalRequired(ApprovalRequired(action_name, arguments, rationale))


def canonicalize_arguments(arguments: dict[str, Any] | None) -> str:
    """Produce a stable JSON representation for approval/action argument binding."""
    return json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
