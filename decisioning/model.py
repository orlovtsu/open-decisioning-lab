from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .synthetic import FEATURES, SyntheticConfig, make_dataset

MODEL_VERSION = "synthetic-logistic-isotonic-1.0"


@dataclass
class RiskModel:
    estimator: LogisticRegression
    calibrator: IsotonicRegression
    metrics: dict[str, float]
    feature_names: tuple[str, ...]

    def predict_probability(self, features: dict[str, float]) -> float:
        row = pd.DataFrame([features], columns=FEATURES)
        raw = float(self.estimator.predict_proba(row)[0, 1])
        calibrated = float(self.calibrator.predict([raw])[0])
        return float(np.clip(calibrated, 0.0, 1.0))

    def explain(self, features: dict[str, float], limit: int = 3) -> list[str]:
        contributions = {
            name: float(coefficient * features[name])
            for name, coefficient in zip(FEATURES, self.estimator.coef_[0])
        }
        ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        return [
            f"{name}:{'increases' if contribution > 0 else 'decreases'}_raw_risk"
            for name, contribution in ranked[:limit]
        ]


def train_model(config: SyntheticConfig = SyntheticConfig()) -> RiskModel:
    frame, target = make_dataset(config)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    train_cutoff = frame["observed_at"].quantile(0.6)
    calibration_cutoff = frame["observed_at"].quantile(0.8)
    train_mask = frame["observed_at"] <= train_cutoff
    calibration_mask = (
        (frame["observed_at"] > train_cutoff)
        & (frame["observed_at"] <= calibration_cutoff)
    )
    holdout_mask = frame["observed_at"] > calibration_cutoff

    estimator = LogisticRegression(max_iter=1000, random_state=config.seed)
    estimator.fit(frame.loc[train_mask, FEATURES], target.loc[train_mask])

    calibration_raw = estimator.predict_proba(frame.loc[calibration_mask, FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw, target.loc[calibration_mask])

    holdout_raw = estimator.predict_proba(frame.loc[holdout_mask, FEATURES])[:, 1]
    holdout_probability = calibrator.predict(holdout_raw)
    metrics = {
        "holdout_roc_auc": float(roc_auc_score(target.loc[holdout_mask], holdout_raw)),
        "holdout_brier_score": float(
            brier_score_loss(target.loc[holdout_mask], holdout_probability)
        ),
    }
    return RiskModel(estimator, calibrator, metrics, tuple(FEATURES))


_MODEL = train_model()


def get_model() -> RiskModel:
    return _MODEL
