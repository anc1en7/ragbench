"""OpenTelemetry stage tracing, off unless ``TRACING=1`` (Task 5.2)."""

from ragbench.tracing.setup import (
    configure_tracing,
    current_trace_id,
    request_span,
    stage_span,
    tracing_enabled,
)

__all__ = [
    "configure_tracing",
    "current_trace_id",
    "request_span",
    "stage_span",
    "tracing_enabled",
]
