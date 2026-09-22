from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request body for a single prediction.
    """

    text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
        description="News article text",
    )


class PredictionResponse(BaseModel):
    """
    Response returned by the prediction endpoint.
    """

    prediction: str

    label_id: int

    decision_score: float

    model: str

    feature_type: str

    request_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of news articles",
    )


class BatchPredictionResponse(BaseModel):

    results: List[dict]

    count: int

    request_id: Optional[str] = None


class HealthResponse(BaseModel):

    status: str

    model_ready: bool


class ModelInfoResponse(BaseModel):

    model: str

    feature_type: str

    prediction_labels: dict

    warning: str
