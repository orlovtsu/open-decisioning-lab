from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["approve", "review", "decline"]


class DecisionRequest(BaseModel):
    entity_id: str = Field(min_length=1, max_length=100)
    observed_at: date
    monthly_income: float = Field(ge=0)
    account_age_days: int = Field(ge=0)
    recent_event_count: int = Field(ge=0)
    balance_volatility: float = Field(ge=0, le=10)
    identity_confidence: float = Field(ge=0, le=1)
    evidence_completeness: float = Field(ge=0, le=1)


class DecisionResponse(BaseModel):
    decision: Decision
    risk_probability: float = Field(ge=0, le=1)
    reasons: list[str]
    model_reasons: list[str]
    model_version: str
    policy_version: str
    evidence_complete: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    features: list[str]
    metrics: dict[str, float]
