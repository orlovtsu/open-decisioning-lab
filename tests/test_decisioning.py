from datetime import date

from fastapi.testclient import TestClient

from decisioning.api import app
from decisioning.schemas import DecisionRequest
from decisioning.service import evaluate

client = TestClient(app)


def test_synthetic_model_returns_probability_and_versions():
    result = evaluate(
        DecisionRequest(
            entity_id="demo-001",
            observed_at=date(2026, 1, 15),
            monthly_income=4200,
            account_age_days=240,
            recent_event_count=1,
            balance_volatility=0.18,
            identity_confidence=0.96,
            evidence_completeness=0.94,
        )
    )
    assert 0 <= result.risk_probability <= 1
    assert result.model_version.startswith("synthetic-")
    assert result.policy_version == "policy-1.0"
    assert len(result.model_reasons) == 3


def test_incomplete_evidence_routes_to_review():
    response = client.post(
        "/evaluate",
        json={
            "entity_id": "demo-002",
            "observed_at": "2026-01-15",
            "monthly_income": 4200,
            "account_age_days": 240,
            "recent_event_count": 0,
            "balance_volatility": 0.1,
            "identity_confidence": 0.98,
            "evidence_completeness": 0.4,
        },
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "review"
    assert response.json()["reasons"] == ["incomplete_evidence"]
    assert response.json()["model_reasons"]


def test_policy_thresholds_are_ordered_and_explicit():
    from decisioning.policy import apply_policy

    assert apply_policy(0.10, identity_confidence=0.9, evidence_completeness=0.9).decision == "approve"
    assert apply_policy(0.15, identity_confidence=0.9, evidence_completeness=0.9).decision == "review"
    assert apply_policy(0.25, identity_confidence=0.9, evidence_completeness=0.9).decision == "decline"


def test_policy_configuration_changes_outcome_without_changing_model_score():
    from decisioning.model import get_model
    from decisioning.policy import PolicyConfig, apply_policy

    features = {
        "monthly_income": 4200,
        "account_age_days": 240,
        "recent_event_count": 1,
        "balance_volatility": 0.18,
        "identity_confidence": 0.96,
        "evidence_completeness": 0.94,
    }
    score = get_model().predict_probability(features)
    default_result = apply_policy(score, identity_confidence=0.96, evidence_completeness=0.94)
    strict_result = apply_policy(
        score,
        identity_confidence=0.96,
        evidence_completeness=0.94,
        config=PolicyConfig(version="policy-2.0", evidence_floor=0.99),
    )
    assert get_model().predict_probability(features) == score
    assert strict_result.decision != default_result.decision


def test_holdout_report_contains_threshold_and_cohort_views():
    from decisioning.evaluation import build_holdout_report
    from decisioning.model import get_model

    report = build_holdout_report(get_model())
    assert 0 <= report.overall["roc_auc"] <= 1
    assert report.overall["brier_score"] >= 0
    assert len(report.thresholds) == 4
    assert len(report.cohorts) == 2
    assert {row["cohort"] for row in report.cohorts} == {
        "stable_synthetic_cohort",
        "volatile_synthetic_cohort",
    }


def test_synthetic_dataset_is_reproducible_and_quality_checked():
    import numpy as np

    from decisioning.synthetic import FEATURES, SyntheticConfig, make_dataset

    first_frame, first_target = make_dataset(SyntheticConfig(seed=7, rows=1000))
    second_frame, second_target = make_dataset(SyntheticConfig(seed=7, rows=1000))
    assert first_frame.equals(second_frame)
    assert first_target.equals(second_target)
    assert first_frame[FEATURES].notna().all().all()
    assert np.isfinite(first_frame[FEATURES].to_numpy()).all()
    assert first_frame["event" if "event" in first_frame else "latent_event_probability"].notna().all()
    assert 0.05 < first_target.mean() < 0.95
    assert set(first_frame["data_quality_band"]) <= {"low", "medium", "high"}
    assert set(first_frame["synthetic_cohort"]) == {
        "stable_synthetic_cohort",
        "volatile_synthetic_cohort",
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_exposes_evaluation_contract():
    response = client.get("/model-info")
    body = response.json()
    assert response.status_code == 200
    assert body["model_version"].startswith("synthetic-")
    assert body["features"]
    assert 0 <= body["metrics"]["holdout_roc_auc"] <= 1
    assert body["metrics"]["holdout_brier_score"] >= 0
