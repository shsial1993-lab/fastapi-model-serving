from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ClassifierService


service = ClassifierService()
app = FastAPI(title='FastAPI Model Serving Template', version='1.0.0')


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    predicted_class: int
    probabilities: list[float]


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/metadata')
def metadata() -> dict[str, object]:
    return service.metadata


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        label, probabilities = service.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(predicted_class=label, probabilities=probabilities)
