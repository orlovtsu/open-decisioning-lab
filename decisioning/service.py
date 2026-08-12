from .model import MODEL_VERSION, get_model
from .policy import POLICY_VERSION, apply_policy
from .schemas import DecisionRequest, DecisionResponse


def evaluate(request: DecisionRequest) -> DecisionResponse:
    features = request.model_dump(exclude={"entity_id", "observed_at"})
    risk_probability = get_model().predict_probability(features)
    policy_result = apply_policy(
        risk_probability,
        identity_confidence=request.identity_confidence,
        evidence_completeness=request.evidence_completeness,
    )
    model_reasons = get_model().explain(features)
    return DecisionResponse(
        decision=policy_result.decision,
        risk_probability=round(risk_probability, 6),
        reasons=policy_result.reasons,
        model_reasons=model_reasons,
        model_version=MODEL_VERSION,
        policy_version=POLICY_VERSION,
        evidence_complete=policy_result.evidence_complete,
    )
