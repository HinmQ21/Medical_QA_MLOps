"""Prometheus metrics for the API and retrieval service."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Real model backends (llama.cpp CPU, 3B DGX) can take 30-120s; the prometheus default
# buckets cap at 10s, which saturates +Inf and breaks p95/alerting. Use wide buckets.
LATENCY_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300)

REQUEST_COUNT = Counter(
    "mqa_requests_total",
    "Total prediction requests.",
    ["endpoint", "backend", "status"],
)
REQUEST_LATENCY = Histogram(
    "mqa_request_latency_seconds",
    "Request latency in seconds.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
RETRIEVAL_LATENCY = Histogram(
    "mqa_retrieval_latency_seconds",
    "Retrieval latency in seconds.",
    buckets=LATENCY_BUCKETS,
)
RETRIEVAL_NO_RESULT = Counter(
    "mqa_retrieval_no_result_total",
    "Retrieval calls that returned zero evidence.",
)
MODEL_LATENCY = Histogram(
    "mqa_model_latency_seconds",
    "Time spent inside model backend.chat() calls per request.",
    ["backend"],
    buckets=LATENCY_BUCKETS,
)


def observe_request(endpoint: str, backend: str, status: str, latency_s: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, backend=backend, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def observe_retrieval(latency_s: float, no_result: bool) -> None:
    RETRIEVAL_LATENCY.observe(latency_s)
    if no_result:
        RETRIEVAL_NO_RESULT.inc()


def observe_model(backend: str, latency_s: float) -> None:
    MODEL_LATENCY.labels(backend=backend).observe(latency_s)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
