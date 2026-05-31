"""Prometheus metrics for the API and retrieval service."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "mqa_requests_total",
    "Total prediction requests.",
    ["endpoint", "backend", "status"],
)
REQUEST_LATENCY = Histogram(
    "mqa_request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
)
RETRIEVAL_LATENCY = Histogram(
    "mqa_retrieval_latency_seconds",
    "Retrieval latency in seconds.",
)
RETRIEVAL_NO_RESULT = Counter(
    "mqa_retrieval_no_result_total",
    "Retrieval calls that returned zero evidence.",
)


def observe_request(endpoint: str, backend: str, status: str, latency_s: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, backend=backend, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def observe_retrieval(latency_s: float, no_result: bool) -> None:
    RETRIEVAL_LATENCY.observe(latency_s)
    if no_result:
        RETRIEVAL_NO_RESULT.inc()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
