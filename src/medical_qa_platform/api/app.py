"""FastAPI pre/post-processing API."""

import time
import uuid

from fastapi import FastAPI, Response

from ..config import Settings
from ..drift.collector import DriftCollector
from ..observability.metrics import observe_request, render_metrics
from ..retrieval.backends import SupportsSearch
from ..retrieval.contract import RETRIEVAL_CONTRACT_VERSION
from .parser import parse_answer
from .prompt import build_prompt
from .schemas import PredictRequest, PredictResponse


def create_app(
    backend=None,
    retrieval: SupportsSearch | None = None,
    model_version: str | None = None,
    drift_log_path: str | None = None,
    top_k: int | None = None,
) -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(title="Medical QA API")
    app.state.top_k = top_k if top_k is not None else settings.top_k
    app.state.model_version = (
        model_version if model_version is not None else settings.model_version
    )
    app.state.collector = DriftCollector(
        drift_log_path if drift_log_path is not None else settings.drift_log_path
    )
    if backend is not None:
        app.state.backend = backend
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if getattr(app.state, "backend", None) is None:
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        trace_id = uuid.uuid4().hex
        evidence = app.state.retrieval.search(req.question, app.state.top_k)
        messages = build_prompt(req.question, req.options, evidence)
        raw = app.state.backend.generate(messages)
        answer = parse_answer(raw, valid_letters=set(req.options))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp = PredictResponse(
            answer=answer,
            evidence=evidence,
            backend=app.state.backend.name,
            model_version=app.state.model_version,
            contract_version=RETRIEVAL_CONTRACT_VERSION,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
        observe_request(
            endpoint="/predict",
            backend=app.state.backend.name,
            status="ok" if answer is not None else "no_answer",
            latency_s=latency_ms / 1000.0,
        )
        app.state.collector.record(req, resp, n_evidence=len(evidence))
        return resp

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
