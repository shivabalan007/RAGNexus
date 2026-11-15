from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from ragnexus.config import settings
from ragnexus.observability.context import request_id_var, trace_id_var


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        trace_id = trace_id_var.get()
        if trace_id == "-":
            try:
                from opentelemetry import trace
                span_context = trace.get_current_span().get_span_context()
                if span_context.is_valid:
                    trace_id = f"{span_context.trace_id:032x}"
            except ImportError:
                pass
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id,
        }
        for key in ("workspace_id", "session_id", "pipeline", "stage", "latency_ms", "error_code", "worker_id", "job_id", "document_id", "attempt"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
