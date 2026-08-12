# Model Card

## Scope

This model is a synthetic demonstration of calibrated binary risk estimation. It must not be used to make real decisions about people, access, eligibility, pricing, employment, housing, healthcare, or financial services.

## Data

The training data is generated locally from seeded random distributions. It includes temporal drift, seasonality, correlated synthetic cohorts, data-quality bands, and a latent interaction between recent events and volatility. It does not contain real people, organizations, accounts, transactions, or external records.

## Evaluation

The pipeline uses three chronological periods: training, calibration, and final holdout. A logistic estimator is fitted on the training period, an isotonic calibrator is fitted on the calibration period, and ROC AUC plus Brier score are reported only on the untouched final holdout period.

## Limitations

Synthetic relationships do not represent a real population. Metrics are illustrative, not evidence of production performance. Thresholds are demonstration values and require domain, legal, fairness, and safety review in any real application.

## Explanations

The API returns the strongest signed linear contributions as `model_reasons`. They explain direction in the uncalibrated model score, not causal effects and not a guarantee about the calibrated probability. A production system would validate explanation stability and user-facing wording separately.

## Subgroup analysis

The evaluation module compares two synthetic cohorts split by account-history length and reports cohort-level calibration error and observed event rate. These cohorts are artificial diagnostic slices; they are not protected groups and must not be interpreted as a real-world fairness assessment.
