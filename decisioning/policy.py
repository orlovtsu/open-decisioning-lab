from dataclasses import dataclass

from .schemas import Decision


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reasons: list[str]
    evidence_complete: bool


@dataclass(frozen=True)
class PolicyConfig:
    version: str = "policy-1.0"
    evidence_floor: float = 0.8
    identity_floor: float = 0.7
    review_threshold: float = 0.12
    decline_threshold: float = 0.20


POLICY_VERSION = "policy-1.0"


def apply_policy(
    risk_probability: float,
    *,
    identity_confidence: float,
    evidence_completeness: float,
    config: PolicyConfig = PolicyConfig(),
) -> PolicyResult:
    reasons: list[str] = []
    evidence_complete = evidence_completeness >= config.evidence_floor

    if not evidence_complete:
        reasons.append("incomplete_evidence")
        return PolicyResult("review", reasons, False)

    if identity_confidence < config.identity_floor:
        reasons.append("identity_confidence_below_policy_floor")
        return PolicyResult("review", reasons, True)

    if risk_probability >= config.decline_threshold:
        reasons.append("risk_above_decline_threshold")
        return PolicyResult("decline", reasons, True)

    if risk_probability >= config.review_threshold:
        reasons.append("risk_above_review_threshold")
        return PolicyResult("review", reasons, True)

    reasons.append("risk_below_review_threshold")
    return PolicyResult("approve", reasons, True)
