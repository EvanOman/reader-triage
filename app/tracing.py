"""OpenTelemetry tracing setup for Reader Triage."""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "reader-triage"

_sqla_instrumentor = SQLAlchemyInstrumentor()


def setup_tracing(app):
    """Configure OpenTelemetry with OTLP export to Jaeger."""
    resource = Resource.create({"service.name": SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    # Instrument FastAPI (auto-creates spans for all HTTP requests)
    FastAPIInstrumentor.instrument_app(app)

    return provider


def instrument_engine(async_engine):
    """Instrument an async SQLAlchemy engine for tracing."""
    _sqla_instrumentor.instrument(engine=async_engine.sync_engine)
