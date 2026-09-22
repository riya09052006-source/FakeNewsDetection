from datetime import datetime
from typing import List

from pydantic import BaseModel


class PredictionHistoryItem(BaseModel):

    id: int

    prediction: str

    label_id: int

    decision_score: float

    model: str

    feature_type: str

    created_at: datetime


class PredictionHistoryDetail(
    PredictionHistoryItem
):

    text: str


class PredictionHistoryResponse(BaseModel):

    count: int

    items: List[
        PredictionHistoryItem
    ]


class AnalyticsResponse(BaseModel):

    total: int

    fake_count: int

    real_count: int

    fake_percentage: float

    real_percentage: float


class ClearHistoryResponse(BaseModel):

    deleted_count: int

    message: str
