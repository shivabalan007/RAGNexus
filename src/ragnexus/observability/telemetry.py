from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure_opentelemetry(app=None, engine=None) -> None:
    """Enable OTLP traces only when configured; API remains usable without a collector."""
    from ragnexus.config import get_settings
    settings = get_settings()
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        provider = TracerProvider(resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME}))
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT or None, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        if app is not None:
            FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        if engine is not None:
            SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=provider)
    except Exception as exc:
        logger.warning("OpenTelemetry instrumentation unavailable: %s", type(exc).__name__)
