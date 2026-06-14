"""Prometheus metrics for the API and retrieval service."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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
TOOL_OUTCOME = Counter(
    "mqa_tool_outcome_total",
    "Per-request tool usage outcome.",
    ["outcome"],  # not_called | empty | hit
)
TOOL_CALLS_PER_REQUEST = Histogram(
    "mqa_tool_calls_per_request",
    "Number of tool calls the model made in one /predict.",
    buckets=(0, 1, 2, 3, 4, 5),
)
BUILD_INFO = Gauge(
    "mqa_build_info",
    "Deployed version info (value is always 1).",
    ["model_version", "contract_version", "backend"],
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


def observe_tool(tool_call_count: int, outcome: str) -> None:
    TOOL_OUTCOME.labels(outcome=outcome).inc()
    TOOL_CALLS_PER_REQUEST.observe(tool_call_count)


def set_build_info(model_version: str | None, contract_version: str, backend: str) -> None:
    BUILD_INFO.labels(
        model_version=model_version or "unknown",
        contract_version=contract_version,
        backend=backend,
    ).set(1)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
