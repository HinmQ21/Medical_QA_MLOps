"""Standalone retrieval FastAPI service."""

import os
import time

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from ..observability.metrics import observe_retrieval, render_metrics
from .backends import SupportsSearch
from .contract import RETRIEVAL_CONTRACT_VERSION


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    query: str
    results: list[str]


def _build_backend_from_env() -> SupportsSearch:  # pragma: no cover
    kind = os.environ.get("RETRIEVAL_BACKEND", "kg").lower()
    if kind == "kg":
        from .kg_backend import KGRetrieval

        return KGRetrieval()
    raise ValueError(f"unknown RETRIEVAL_BACKEND: {kind!r}")


def create_retrieval_service(backend: SupportsSearch | None = None) -> FastAPI:
    app = FastAPI(title="Medical Knowledge Retrieval Service")
    if backend is not None:
        app.state.backend = backend

    @app.on_event("startup")
    def _startup() -> None:
        app.state.backend = backend if backend is not None else _build_backend_from_env()

    @app.post("/search", response_model=list[SearchResult])
    def search(req: SearchRequest):
        out = []
        for query in req.queries:
            t0 = time.perf_counter()
            results = app.state.backend.search(query, req.top_k)
            observe_retrieval(time.perf_counter() - t0, no_result=len(results) == 0)
            out.append(SearchResult(query=query, results=results))
        return out

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if getattr(app.state, "backend", None) is None:
            response.status_code = 503
            return {"status": "not ready"}
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics():
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    @app.get("/version")
    def version():
        return {
            "contract_version": RETRIEVAL_CONTRACT_VERSION,
            "encoder_model": os.environ.get(
                "KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1"
            ),
            "kg_data_dir": os.environ.get("KG_DATA_DIR", "data/"),
        }

    return app
