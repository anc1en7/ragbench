"""OpenTelemetry tracing for pipeline stages, exported to Phoenix over OTLP/HTTP.

Tracing is OFF unless the env var ``TRACING`` is ``1``. When off, no tracer provider or
exporter is configured, the OpenTelemetry SDK is never imported, and :func:`stage_span`
returns a shared no-op context manager, so the overhead is a single boolean check.

When on, :func:`configure_tracing` installs a ``TracerProvider`` with a batch OTLP/HTTP
exporter. The endpoint is ``PHOENIX_COLLECTOR_ENDPOINT`` (default
``http://localhost:6006/v1/traces``, the compose ``phoenix`` profile).
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"
SERVICE_NAME = "ragbench"

_NOOP: AbstractContextManager[None] = nullcontext()
_state: dict[str, Any] = {"configured": False, "tracer": None}


def tracing_enabled() -> bool:
    """True when the ``TRACING`` env var is exactly ``1``."""
    return os.environ.get("TRACING", "0").strip() == "1"


def configure_tracing(*, exporter: Any | None = None) -> bool:
    """Install the tracer provider once if tracing is enabled; return whether it is on.

    ``exporter`` is a test seam (e.g. an ``InMemorySpanExporter``); it is wrapped in a
    ``SimpleSpanProcessor`` so tests see spans synchronously. By default a batch OTLP/HTTP
    exporter to Phoenix is used.
    """
    if not tracing_enabled():
        return False
    if _state["configured"]:
        return True
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _state["tracer"] = provider.get_tracer(SERVICE_NAME)
    _state["configured"] = True
    return True


def _reset_for_tests() -> None:
    """Forget the configured tracer (tests only)."""
    _state["configured"] = False
    _state["tracer"] = None


def stage_span(name: str, **attributes: str | int | float) -> AbstractContextManager[Any]:
    """A span named ``ragbench.<name>`` when tracing is on, else a shared no-op."""
    tracer = _state["tracer"]
    if tracer is None:
        return _NOOP
    span_cm: AbstractContextManager[Any] = tracer.start_as_current_span(
        f"ragbench.{name}", attributes=attributes
    )
    return span_cm


@contextmanager
def request_span(name: str, **attributes: str | int | float) -> Iterator[str]:
    """Wrap one request in a root span and yield its trace id (32 hex chars).

    With tracing off, the yielded id is a random 32-hex request id so API responses
    always carry a ``trace_id``.
    """
    tracer = _state["tracer"]
    if tracer is None:
        yield secrets.token_hex(16)
        return
    with tracer.start_as_current_span(f"ragbench.{name}", attributes=attributes):
        yield current_trace_id() or secrets.token_hex(16)


def current_trace_id() -> str | None:
    """Hex trace id of the active span, or ``None`` when tracing is off or no span is active."""
    if _state["tracer"] is None:
        return None
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")
