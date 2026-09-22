from typing import List

from pydantic import BaseModel, Field


class ExplanationFeature(BaseModel):
    feature: str
    contribution: float


class ExplanationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
    )


class ExplanationResponse(BaseModel):
    prediction: str
    label_id: int
    decision_score: float

    top_real_features: List[ExplanationFeature]

    top_fake_features: List[ExplanationFeature]

    explanation_note: str

    request_id: str
