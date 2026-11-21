import pytest
from pydantic import ValidationError

from ragnexus.schemas.chat import ChatRequest


def test_chat_request_defaults_to_standard_pipeline():
    request = ChatRequest(session_id="8e9b4e63-25d1-4f16-b75d-7d55e2d9632d", query="hello")
    assert request.pipeline_mode == "standard"
    assert request.enable_web_search is False


def test_chat_request_accepts_agentic_pipeline_and_opt_in_web_search():
    request = ChatRequest(
        session_id="8e9b4e63-25d1-4f16-b75d-7d55e2d9632d", query="latest release",
        pipeline_mode="agentic", enable_web_search=True,
    )
    assert request.pipeline_mode == "agentic"
    assert request.enable_web_search is True


def test_chat_request_rejects_unknown_pipeline():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="8e9b4e63-25d1-4f16-b75d-7d55e2d9632d", query="hello", pipeline_mode="magic")


def test_chat_request_accepts_idempotency_request_id():
    request = ChatRequest(
        session_id="8e9b4e63-25d1-4f16-b75d-7d55e2d9632d",
        query="hello",
        request_id="2c1a0db7-5a64-4b2f-9d90-7f0c5db7d6b4",
    )
    assert str(request.request_id) == "2c1a0db7-5a64-4b2f-9d90-7f0c5db7d6b4"
