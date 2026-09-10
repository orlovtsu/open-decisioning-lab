# Open Decisioning Lab

A domain-neutral reference implementation of a production decisioning system.

This project combines a calibrated machine-learning model with explicit policy rules, synthetic data, stable reason codes, and a typed HTTP API. It contains no company-specific logic, real customer data, real product names, or production model artifacts.

## Demonstrated capabilities

- deterministic synthetic event generation
- temporal drift, seasonality, cohort correlation, and data-quality metadata
- time-aware train/calibration split
- probability calibration
- separation of model scoring from policy decisions
- stable human-readable reason codes
- conservative handling of incomplete evidence
- typed FastAPI contracts
- unit and API tests
- container and CI configuration

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
uvicorn decisioning.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

`GET /model-info` exposes the synthetic model version, feature contract, and holdout ROC AUC and Brier score used by the local evaluation run.

## Example request

```json
{
  "entity_id": "demo-001",
  "observed_at": "2026-01-15",
  "monthly_income": 4200,
  "account_age_days": 240,
  "recent_event_count": 1,
  "balance_volatility": 0.18,
  "identity_confidence": 0.96,
  "evidence_completeness": 0.94
}
```

The values are illustrative only. This educational project is not intended for real-world eligibility decisions.

## Architecture

```text
synthetic data -> feature builder -> calibrated model -> policy engine -> API response
                                      \-> model reasons
```

The model estimates a synthetic event probability. The policy engine independently maps that estimate and evidence checks to `approve`, `review`, or `decline`. This boundary makes policy changes auditable without silently changing model behavior.

Responses also include signed `model_reasons` for the strongest feature contributions. These are directional explanations of the synthetic model score, not causal claims.

The reusable `decisioning.evaluation` module produces holdout metrics, threshold trade-offs, and comparisons across two explicitly synthetic cohorts. The cohort report is a methodology demonstration, not a fairness claim about real populations.

Synthetic data details are documented in [docs/synthetic-data.md](docs/synthetic-data.md). Build a reproducible JSON report with:

```powershell
python scripts/generate_evaluation_report.py --seed 42 --rows 2400
```

The generated `reports/index.html` is the full analytical dossier. GitHub-native readers can open [reports/REPORT.md](reports/REPORT.md), which embeds the charts directly in Markdown. The report includes methodology, ROC and precision-recall curves, calibration, threshold trade-offs, data-quality profile, synthetic cohort comparison, temporal stability, feature effects, limitations, and the underlying holdout metrics. Separate PNG files are generated for each major analysis.

See [architecture notes](docs/architecture.md) and the [failure-mode register](docs/failure-modes.md) for design trade-offs and production extension points.

## Portfolio note

The repository is intentionally domain-neutral. It demonstrates engineering judgment around calibration, explainability, safety defaults, testing, and operational boundaries without exposing confidential business rules or data.
