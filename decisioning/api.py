from fastapi import FastAPI

from .model import MODEL_VERSION, get_model
from .schemas import DecisionRequest, DecisionResponse, ModelInfoResponse
from .service import evaluate

app = FastAPI(title="Open Decisioning Lab", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = get_model()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        features=[*model.feature_names],
        metrics=model.metrics,
    )


@app.post("/evaluate", response_model=DecisionResponse)
def evaluate_request(request: DecisionRequest) -> DecisionResponse:
    return evaluate(request)
