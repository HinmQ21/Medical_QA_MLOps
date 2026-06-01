"""KServe-compatible mock predictor used for CPU smoke serving."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..inference.mock_backend import MockBackend


class PredictInstance(BaseModel):
    messages: list[dict] = Field(default_factory=list)


class PredictRequest(BaseModel):
    instances: list[PredictInstance]


class Prediction(BaseModel):
    text: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]


def create_app(backend: MockBackend | None = None) -> FastAPI:
    app = FastAPI(title="Medical QA KServe Mock Predictor")
    model = backend or MockBackend()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready"}

    def _predict(req: PredictRequest) -> PredictResponse:
        predictions = [
            Prediction(text=model.generate(instance.messages))
            for instance in req.instances
        ]
        return PredictResponse(predictions=predictions)

    @app.post("/v1/models/{model_name}:predict", response_model=PredictResponse)
    def predict_v1(model_name: str, req: PredictRequest):
        return _predict(req)

    @app.post("/predict", response_model=PredictResponse)
    def predict_short(req: PredictRequest):
        return _predict(req)

    return app
