from typing import List

from pydantic import BaseModel
from pydantic import Field


class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
    )


class PredictionResponse(BaseModel):

    prediction: str

    label_id: int

    decision_score: float

    model: str

    feature_type: str

    request_id: str


class BatchPredictionRequest(BaseModel):

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
    )


class BatchPredictionResponse(BaseModel):

    results: list

    count: int

    request_id: str


class HealthResponse(BaseModel):

    status: str

    model_ready: bool


class ModelInfoResponse(BaseModel):

    model: str

    feature_type: str

    prediction_labels: dict

    warning: str
