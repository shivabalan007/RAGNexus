import requests
import pytest

from ui.api_client import APIError, RAGNexusAPI


class TimeoutSession:
    def request(self, *_args, **_kwargs):
        raise requests.Timeout("read timed out")


def test_chat_api_timeout_is_identified_for_recovery():
    client = RAGNexusAPI("http://127.0.0.1:8000/api/v1")
    client.session = TimeoutSession()

    with pytest.raises(APIError) as captured:
        client.send_message(
            "session-id",
            "question",
            request_id="2c1a0db7-5a64-4b2f-9d90-7f0c5db7d6b4",
            timeout=600,
        )

    assert captured.value.timed_out is True
    assert captured.value.status_code is None
