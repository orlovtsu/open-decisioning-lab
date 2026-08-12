from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURES = [
    "monthly_income",
    "account_age_days",
    "recent_event_count",
    "balance_volatility",
    "identity_confidence",
    "evidence_completeness",
]


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    rows: int = 2400
    start_date: str = "2024-01-01"


def make_dataset(config: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start_date, periods=config.rows, freq="D")
    time_index = np.arange(config.rows) / max(config.rows - 1, 1)
    month_of_year = dates.month.to_numpy()
    cohort = rng.choice(
        ["stable_synthetic_cohort", "volatile_synthetic_cohort"],
        size=config.rows,
        p=[0.72, 0.28],
    )
    volatile = cohort == "volatile_synthetic_cohort"

    income_trend = 1 + 0.12 * time_index
    monthly_income = rng.lognormal(mean=8.05, sigma=0.38, size=config.rows) * income_trend
    monthly_income *= np.where(volatile, 0.86, 1.0)
    account_age_days = np.clip(rng.gamma(shape=2.8, scale=115, size=config.rows), 5, 900).astype(int)
    seasonal_activity = 1 + 0.22 * np.isin(month_of_year, [1, 6, 12])
    event_rate = 0.85 + 0.35 * volatile + 0.25 * seasonal_activity
    recent_event_count = rng.poisson(event_rate)
    balance_volatility = np.clip(
        rng.gamma(shape=2.0, scale=np.where(volatile, 0.34, 0.18)), 0, 3
    )
    identity_confidence = np.clip(
        rng.beta(np.where(volatile, 8, 13), np.where(volatile, 2.5, 1.7)), 0, 1
    )
    evidence_completeness = np.clip(
        rng.beta(np.where(volatile, 8, 15), np.where(volatile, 2.5, 1.8)), 0, 1
    )
    data_quality_band = np.select(
        [evidence_completeness < 0.7, evidence_completeness < 0.85],
        ["low", "medium"],
        default="high",
    )

    logit = (
        -0.6
        - 0.00008 * monthly_income
        - 0.0012 * account_age_days
        + 0.26 * recent_event_count
        + 0.72 * balance_volatility
        + 0.10 * recent_event_count * balance_volatility
        - 1.4 * identity_confidence
        - 1.1 * evidence_completeness
        + 0.18 * time_index
        + rng.normal(0, 0.35, size=config.rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "observed_at": dates,
            "monthly_income": monthly_income,
            "account_age_days": account_age_days,
            "recent_event_count": recent_event_count,
            "balance_volatility": balance_volatility,
            "identity_confidence": identity_confidence,
            "evidence_completeness": evidence_completeness,
            "synthetic_cohort": cohort,
            "data_quality_band": data_quality_band,
            "latent_event_probability": probability,
        }
    )
    return frame, pd.Series(outcome, name="event")
