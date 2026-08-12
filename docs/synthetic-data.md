# Synthetic Data Contract

The generator is deterministic for a given `seed` and creates only artificial observations.

## Generated structure

- `observed_at`: ordered daily observations with a configurable start date;
- `monthly_income`: positive log-normal values with a small temporal trend;
- `account_age_days`: bounded gamma-distributed history length;
- `recent_event_count`: seasonal Poisson activity with cohort-specific intensity;
- `balance_volatility`: bounded gamma values with higher variance in the volatile cohort;
- `identity_confidence`: bounded beta-distributed confidence;
- `evidence_completeness`: bounded beta-distributed data quality;
- `synthetic_cohort`: stable or volatile artificial cohort label;
- `data_quality_band`: low, medium, or high diagnostic label;
- `latent_event_probability`: hidden generator probability used only for diagnostics;
- `event`: sampled binary target returned separately from the feature frame.

The model uses only the six names in `FEATURES`. Diagnostic metadata is intentionally excluded from model training to prevent leakage.

## Why this is useful

The generator has enough structure to exercise temporal validation, calibration, cohort comparisons, missing-evidence policy, and threshold analysis. It is still not a substitute for representative real-world data, and no metric from this dataset should be interpreted as production evidence.
