from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ragnexus.observability.context import request_id_var, trace_id_var


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        supplied = request.headers.get("x-request-id", "").strip()
        request_id = supplied[:128] if supplied and all(ch.isalnum() or ch in "-_." for ch in supplied) else str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        try:
            from opentelemetry import trace
            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                trace_id = f"{span_context.trace_id:032x}"
        except ImportError:
            pass
        request_token = request_id_var.set(request_id)
        trace_token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            request_id_var.reset(request_token)
            trace_id_var.reset(trace_token)
