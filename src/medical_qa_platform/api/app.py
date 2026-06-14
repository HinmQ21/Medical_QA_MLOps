"""FastAPI pre/post-processing API."""

import time
import uuid

from fastapi import FastAPI, HTTPException, Response

from ..config import Settings
from ..drift.collector import DriftCollector
from ..inference.agent import run_agentic_loop
from ..observability.logging import configure_logging, get_logger
from ..observability.metrics import (
    observe_model,
    observe_request,
    observe_tool,
    render_metrics,
    set_build_info,
)
from ..retrieval.backends import SupportsSearch
from ..retrieval.contract import RETRIEVAL_CONTRACT_VERSION
from .parser import parse_answer
from .schemas import PredictRequest, PredictResponse, Turn

logger = get_logger("medical_qa_platform.api")


def create_app(
    backend=None,
    retrieval: SupportsSearch | None = None,
    model_version: str | None = None,
    drift_log_path: str | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    max_tool_iterations: int | None = None,
) -> FastAPI:
    configure_logging()
    settings = Settings.from_env()
    app = FastAPI(title="Medical QA API")
    app.state.top_k = top_k if top_k is not None else settings.top_k
    app.state.max_tokens = (
        max_tokens if max_tokens is not None else settings.max_tokens
    )
    app.state.max_tool_iterations = (
        max_tool_iterations
        if max_tool_iterations is not None
        else settings.max_tool_iterations
    )
    app.state.model_version = (
        model_version if model_version is not None else settings.model_version
    )
    app.state.collector = DriftCollector(
        drift_log_path if drift_log_path is not None else settings.drift_log_path
    )
    if backend is not None:
        app.state.backend = backend
        set_build_info(app.state.model_version, RETRIEVAL_CONTRACT_VERSION, backend.name)
    if retrieval is not None:
        app.state.retrieval = retrieval

    @app.on_event("startup")
    def _startup() -> None:
        if backend is not None:
            app.state.backend = backend
        else:
            from ..inference import get_backend

            app.state.backend = get_backend(settings.model_backend)
        if retrieval is not None:
            app.state.retrieval = retrieval
        else:
            from ..retrieval.client import RetrievalClient

            app.state.retrieval = RetrievalClient.from_env()
        set_build_info(
            app.state.model_version,
            RETRIEVAL_CONTRACT_VERSION,
            app.state.backend.name,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        backend = getattr(app.state, "backend", None)
        if backend is None or not backend.health_check():
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        try:
            result = run_agentic_loop(
                app.state.backend,
                app.state.retrieval,
                req.question,
                top_k=app.state.top_k,
                max_tokens=app.state.max_tokens,
                max_iterations=app.state.max_tool_iterations,
            )
            answer = parse_answer(result.final_content)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp = PredictResponse(
                answer=answer,
                raw_output=result.final_content,
                evidence=result.evidence,
                trace=[Turn(**t) for t in result.trace],
                backend=app.state.backend.name,
                model_version=app.state.model_version,
                contract_version=RETRIEVAL_CONTRACT_VERSION,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
            if result.tool_call_count == 0:
                outcome = "not_called"
            elif not result.evidence:
                outcome = "empty"
            else:
                outcome = "hit"
            status = "ok" if answer is not None else "no_answer"
            observe_request(
                endpoint="/predict",
                backend=app.state.backend.name,
                status=status,
                latency_s=latency_ms / 1000.0,
            )
            observe_model(app.state.backend.name, result.model_latency_s)
            observe_tool(result.tool_call_count, outcome)
            app.state.collector.record(
                req, resp, n_evidence=len(result.evidence), tool_call_count=result.tool_call_count
            )
            logger.info(
                "prediction",
                extra={
                    "trace_id": trace_id,
                    "latency_ms": latency_ms,
                    "backend": app.state.backend.name,
                    "tool_call_count": result.tool_call_count,
                    "n_evidence": len(result.evidence),
                    "status": status,
                },
            )
            return resp
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            # Guard the backend lookup the same way /ready does: if the failure happened
            # before the backend was set, app.state.backend may be absent — don't let the
            # error handler raise its own AttributeError and mask the original exception.
            backend_name = getattr(getattr(app.state, "backend", None), "name", "unknown")
            observe_request(
                endpoint="/predict",
                backend=backend_name,
                status="error",
                latency_s=latency_ms / 1000.0,
            )
            logger.exception("prediction_failed", extra={"trace_id": trace_id})
            raise HTTPException(status_code=500, detail="prediction failed")

    @app.get("/metrics")
    def metrics():
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    @app.get("/version")
    def version():
        backend = getattr(app.state, "backend", None)
        return {
            "contract_version": RETRIEVAL_CONTRACT_VERSION,
            "model_version": app.state.model_version,
            "backend": backend.name if backend is not None else None,
        }

    return app
