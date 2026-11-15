from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def start_span(name: str, attributes: dict | None = None) -> Iterator[object | None]:
    """Create an OpenTelemetry span when the SDK/API is installed; otherwise no-op."""
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return

    tracer = trace.get_tracer("ragnexus")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                try:
                    span.set_attribute(key, value)
                except Exception:
                    # Telemetry must never break retrieval or generation.
                    pass
        yield span
