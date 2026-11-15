import pytest

from ragnexus.agentic.approvals.gate import HumanApprovalRequired, canonicalize_arguments, enforce_tool_approval


def test_non_sensitive_read_only_tool_does_not_require_approval():
    enforce_tool_approval("web.search", {"query": "weather"}, sensitive=False)


def test_sensitive_tool_requires_approval():
    with pytest.raises(HumanApprovalRequired) as exc:
        enforce_tool_approval("documents.delete", {"document_id": "doc-1"}, sensitive=True)
    assert exc.value.request.action_name == "documents.delete"


def test_sensitive_tool_requires_approved_status():
    with pytest.raises(HumanApprovalRequired):
        enforce_tool_approval("documents.delete", {}, sensitive=True, approval_status="pending")
    enforce_tool_approval("documents.delete", {}, sensitive=True, approval_status="approved")


def test_approval_arguments_are_canonicalized_for_exact_matching():
    assert canonicalize_arguments({"document_id": "abc", "force": False}) == canonicalize_arguments(
        {"force": False, "document_id": "abc"}
    )
    assert canonicalize_arguments({"document_id": "abc"}) != canonicalize_arguments(
        {"document_id": "other"}
    )
