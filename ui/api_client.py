from __future__ import annotations

from typing import Any
import requests


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RAGNexusAPI:
    def __init__(self, base_url: str, token: str | None = None, timeout: int = 45):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.token = token

    def set_token(self, token: str | None):
        self.token = token

    def _headers(self):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method, url, headers=self._headers(),
                timeout=kwargs.pop("timeout", self.timeout), **kwargs
            )
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach RAGNexus API at {self.base_url}: {type(exc).__name__}") from exc

        if not response.ok:
            try:
                payload = response.json()
                detail = payload.get("detail", payload)
            except ValueError:
                detail = response.text[:500] or response.reason
            raise APIError(str(detail), response.status_code)

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIError("API returned a non-JSON response", response.status_code) from exc

    def health(self):
        return self._request("GET", "/health", timeout=8)

    def register(self, email: str, password: str):
        return self._request("POST", "/auth/register", json={"email": email, "password": password})

    def login(self, email: str, password: str):
        return self._request("POST", "/auth/login", json={"email": email, "password": password})

    def me(self):
        return self._request("GET", "/auth/me")

    def workspaces(self):
        return self._request("GET", "/workspaces")

    def create_workspace(self, name: str):
        return self._request("POST", "/workspaces", json={"name": name})

    def documents(self, workspace_id: str):
        return self._request("GET", f"/workspaces/{workspace_id}/documents")

    def upload(self, workspace_id: str, filename: str, content: bytes, mime_type: str | None):
        files = {"file": (filename, content, mime_type or "application/octet-stream")}
        return self._request("POST", f"/workspaces/{workspace_id}/documents", files=files, timeout=120)

    def ingestion_job(self, job_id: str):
        return self._request("GET", f"/ingestion/jobs/{job_id}")

    def create_chat_session(
        self, workspace_id: str, visibility: str = "private",
        document_scope: str = "workspace", document_ids: list[str] | None = None,
        title: str | None = None,
    ):
        return self._request(
            "POST", "/chat/sessions",
            json={
                "workspace_id": workspace_id,
                "visibility": visibility,
                "document_scope": document_scope,
                "document_ids": document_ids or [],
                "title": title,
            },
        )

    def chat_messages(self, session_id: str):
        return self._request("GET", f"/chat/sessions/{session_id}/messages")

    def send_message(self, session_id: str, query: str, candidate_k: int = 20, final_k: int = 5):
        return self._request(
            "POST", "/chat/messages",
            json={
                "session_id": session_id,
                "query": query,
                "candidate_k": candidate_k,
                "final_k": final_k,
            },
            timeout=120,
        )
