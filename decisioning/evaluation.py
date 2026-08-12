from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .model import RiskModel
from .synthetic import FEATURES, SyntheticConfig, make_dataset


@dataclass(frozen=True)
class EvaluationReport:
    overall: dict[str, float]
    thresholds: list[dict[str, float]]
    cohorts: list[dict[str, float | str]]


def _split_holdout(frame: pd.DataFrame) -> pd.Series:
    calibration_cutoff = frame["observed_at"].quantile(0.8)
    return frame["observed_at"] > calibration_cutoff


def build_holdout_report(
    model: RiskModel,
    config: SyntheticConfig = SyntheticConfig(),
) -> EvaluationReport:
    frame, target = make_dataset(config)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    holdout = _split_holdout(frame)
    raw_probability = model.estimator.predict_proba(frame.loc[holdout, FEATURES])[:, 1]
    probability = model.calibrator.predict(raw_probability)
    actual = target.loc[holdout].to_numpy()

    overall = {
        "roc_auc": float(roc_auc_score(actual, raw_probability)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "rows": float(len(actual)),
    }

    thresholds = []
    for threshold in (0.05, 0.12, 0.20, 0.30):
        predicted = probability >= threshold
        thresholds.append(
            {
                "threshold": threshold,
                "positive_rate": float(predicted.mean()),
                "true_positive_rate": float(actual[predicted].mean()) if predicted.any() else 0.0,
            }
        )

    holdout_frame = frame.loc[holdout].copy()
    holdout_frame["probability"] = probability
    holdout_frame["actual"] = actual
    holdout_frame["cohort"] = holdout_frame["synthetic_cohort"]
    cohorts = []
    for cohort, group in holdout_frame.groupby("cohort", sort=True):
        cohorts.append(
            {
                "cohort": cohort,
                "rows": float(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "observed_event_rate": float(group["actual"].mean()),
                "brier_score": float(
                    brier_score_loss(group["actual"], group["probability"])
                ),
            }
        )

    return EvaluationReport(overall=overall, thresholds=thresholds, cohorts=cohorts)
