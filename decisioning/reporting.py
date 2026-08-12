from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .model import RiskModel
from .synthetic import FEATURES, SyntheticConfig, make_dataset


def _holdout_predictions(model: RiskModel, config: SyntheticConfig):
    frame, target = make_dataset(config)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"])
    cutoff = frame["observed_at"].quantile(0.8)
    holdout = frame["observed_at"] > cutoff
    raw = model.estimator.predict_proba(frame.loc[holdout, FEATURES])[:, 1]
    calibrated = model.calibrator.predict(raw)
    return frame.loc[holdout].copy(), target.loc[holdout].to_numpy(), raw, calibrated


def build_visual_report(
    model: RiskModel,
    config: SyntheticConfig = SyntheticConfig(),
    output_dir: Path = Path("reports"),
) -> dict[str, float | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, actual, raw, probability = _holdout_predictions(model, config)
    scored = frame.copy()
    scored["actual"] = actual
    scored["raw_probability"] = raw
    scored["probability"] = probability
    metrics = {
        "roc_auc": float(roc_auc_score(actual, raw)),
        "average_precision": float(average_precision_score(actual, raw)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "holdout_rows": float(len(actual)),
        "holdout_event_rate": float(actual.mean()),
    }

    fpr, tpr, _ = roc_curve(actual, raw)
    precision, recall, _ = precision_recall_curve(actual, raw)
    fraction, observed = calibration_curve(actual, probability, n_bins=8, strategy="quantile")
    thresholds = np.array([0.05, 0.12, 0.20, 0.30])
    positive_rates = [(probability >= threshold).mean() for threshold in thresholds]
    observed_rates = [
        actual[probability >= threshold].mean() if (probability >= threshold).any() else 0
        for threshold in thresholds
    ]

    threshold_rows = []
    for threshold, positive_rate, observed_rate in zip(thresholds, positive_rates, observed_rates):
        selected = probability >= threshold
        threshold_rows.append(
            {
                "threshold": threshold,
                "selected_rate": positive_rate,
                "observed_event_rate": observed_rate,
                "precision": float(actual[selected].mean()) if selected.any() else 0.0,
                "recall": float(actual[selected].sum() / actual.sum()) if actual.sum() else 0.0,
            }
        )

    cohort_rows = []
    for cohort, group in scored.groupby("synthetic_cohort", sort=True):
        cohort_rows.append(
            {
                "cohort": cohort,
                "rows": len(group),
                "mean_probability": group["probability"].mean(),
                "observed_event_rate": group["actual"].mean(),
                "brier_score": brier_score_loss(group["actual"], group["probability"]),
            }
        )

    scored["time_quartile"] = pd.qcut(scored["observed_at"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    time_summary = scored.groupby("time_quartile", observed=False)[["probability", "actual"]].mean()

    def save_plot(name: str, title: str, plotter) -> None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        plotter(axis)
        axis.set_title(title)
        axis.grid(alpha=0.2)
        figure.savefig(output_dir / name, dpi=160)
        plt.close(figure)

    save_plot(
        "roc_curve.png",
        "Holdout ROC curve",
        lambda axis: (
            axis.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}"),
            axis.plot([0, 1], [0, 1], "--", color="grey", label="random baseline"),
            axis.set(xlabel="False positive rate", ylabel="True positive rate"),
            axis.legend(),
        ),
    )
    save_plot(
        "precision_recall_curve.png",
        "Holdout precision-recall curve",
        lambda axis: (
            axis.plot(recall, precision, label=f"Average precision = {metrics['average_precision']:.3f}"),
            axis.axhline(actual.mean(), linestyle="--", color="grey", label="event-rate baseline"),
            axis.set(xlabel="Recall", ylabel="Precision"),
            axis.legend(),
        ),
    )
    save_plot(
        "calibration_curve.png",
        "Probability calibration",
        lambda axis: (
            axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration"),
            axis.plot(fraction, observed, "o-", label=f"Brier score = {metrics['brier_score']:.3f}"),
            axis.set(xlabel="Mean predicted probability", ylabel="Observed event rate"),
            axis.legend(),
        ),
    )

    def plot_thresholds(axis):
        axis.plot(thresholds, positive_rates, "o-", label="selected rate")
        axis.plot(thresholds, observed_rates, "o-", label="observed event rate")
        axis.set(xlabel="Threshold", ylabel="Rate", ylim=(0, 1))
        axis.legend()

    save_plot("threshold_tradeoff.png", "Threshold trade-off", plot_thresholds)

    def plot_cohorts(axis):
        labels = [row["cohort"].replace("_synthetic_cohort", "") for row in cohort_rows]
        positions = np.arange(len(labels))
        width = 0.35
        axis.bar(positions - width / 2, [row["mean_probability"] for row in cohort_rows], width, label="predicted")
        axis.bar(positions + width / 2, [row["observed_event_rate"] for row in cohort_rows], width, label="observed")
        axis.set_xticks(positions, labels, rotation=15)
        axis.set(ylabel="Rate")
        axis.legend()

    save_plot("cohort_comparison.png", "Synthetic cohort comparison", plot_cohorts)

    def plot_time(axis):
        axis.plot(time_summary.index.astype(str), time_summary["probability"], "o-", label="predicted")
        axis.plot(time_summary.index.astype(str), time_summary["actual"], "o-", label="observed")
        axis.set(xlabel="Holdout time quartile", ylabel="Rate")
        axis.legend()

    save_plot("time_stability.png", "Temporal stability in holdout", plot_time)

    coefficients = pd.DataFrame(
        {"feature": FEATURES, "coefficient": model.estimator.coef_[0]}
    ).sort_values("coefficient")

    def plot_features(axis):
        colors = ["#c44e52" if value < 0 else "#4c72b0" for value in coefficients["coefficient"]]
        axis.barh(coefficients["feature"], coefficients["coefficient"], color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Logistic coefficient", ylabel="Feature")

    save_plot("feature_effects.png", "Directional feature effects", plot_features)

    def table(headers, rows):
        head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
        return f"<table><tr>{head}</tr>{body}</table>"

    metric_rows = [(name, f"{value:.6f}") for name, value in metrics.items()]
    threshold_table = table(
        ["Threshold", "Selected rate", "Observed event rate", "Precision", "Recall"],
        [
            (f"{row['threshold']:.2f}", f"{row['selected_rate']:.3f}", f"{row['observed_event_rate']:.3f}", f"{row['precision']:.3f}", f"{row['recall']:.3f}")
            for row in threshold_rows
        ],
    )
    cohort_table = table(
        ["Cohort", "Rows", "Mean probability", "Observed event rate", "Brier score"],
        [
            (row["cohort"], row["rows"], f"{row['mean_probability']:.4f}", f"{row['observed_event_rate']:.4f}", f"{row['brier_score']:.4f}")
            for row in cohort_rows
        ],
    )
    feature_table = table(
        ["Feature", "Coefficient"],
        [(row.feature, f"{row.coefficient:.5f}") for row in coefficients.itertuples()],
    )
    quality_table = table(
        ["Quality band", "Rows", "Share"],
        [
            (band, int(count), f"{count / len(frame):.3f}")
            for band, count in frame["data_quality_band"].value_counts().sort_index().items()
        ],
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Open Decisioning Lab: full evaluation report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;line-height:1.55;color:#20252b}}
nav{{background:#f3f5f7;padding:1rem;border-left:4px solid #4c72b0}} section{{margin:2.5rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
img{{width:100%;border:1px solid #ddd}} table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border:1px solid #ddd;padding:.45rem .7rem;text-align:left}} th{{background:#f3f5f7}}
.metric{{font-size:1.4rem;font-weight:650}} code{{background:#f3f5f7;padding:.15rem .3rem}}</style></head>
<body><h1>Open Decisioning Lab</h1>
<p class="metric">Detailed synthetic ML evaluation report</p>
<nav><strong>Executive summary.</strong> This report evaluates a calibrated binary model on a chronological holdout. It separates ranking quality, probability quality, policy thresholds, cohort behavior, temporal stability, and directional feature effects. All records are synthetic.</nav>
<section><h2>1. Methodology</h2><p>Data is generated with a fixed seed, temporal drift, seasonality, two artificial cohorts, data-quality bands, and a nonlinear interaction. The estimator is trained on the first 60% of time, isotonic calibration uses the next 20%, and all reported metrics use the final 20% holdout. Diagnostic metadata is excluded from model features.</p><p>The model is a logistic classifier. ROC AUC measures ranking, average precision emphasizes the positive class, and Brier score measures probabilistic accuracy after calibration. Policy thresholds are analyzed separately from model fitting.</p></section>
<section><h2>2. Holdout metrics</h2>{table(["Metric", "Value"], metric_rows)}<div class="grid"><img src="roc_curve.png" alt="ROC curve"><img src="precision_recall_curve.png" alt="Precision recall curve"></div></section>
<section><h2>3. Calibration and policy thresholds</h2><p>Calibration asks whether a group predicted at probability 0.20 experiences the event at approximately 20%. Threshold analysis shows how selection volume and observed event rate change without retraining the model.</p><div class="grid"><img src="calibration_curve.png" alt="Calibration curve"><img src="threshold_tradeoff.png" alt="Threshold tradeoff"></div>{threshold_table}</section>
<section><h2>4. Data quality profile</h2><p>Evidence completeness is represented as a synthetic diagnostic variable. It is not treated as a favorable signal when missing: the policy layer can route incomplete evidence to review.</p>{quality_table}</section>
<section><h2>5. Cohort and temporal analysis</h2><p>Cohorts are artificial diagnostic slices, not protected groups and not a real-world fairness assessment. Temporal stability compares predicted and observed rates across holdout time quartiles.</p><div class="grid"><img src="cohort_comparison.png" alt="Cohort comparison"><img src="time_stability.png" alt="Temporal stability"></div>{cohort_table}</section>
<section><h2>6. Feature effects</h2><p>These coefficients show directional effects in the uncalibrated logistic score. They are not causal effects and should not be interpreted as individual explanations without additional validation.</p><img src="feature_effects.png" alt="Feature effects">{feature_table}</section>
<section><h2>7. Limitations and next steps</h2><ul><li>Synthetic distributions do not establish production validity.</li><li>Cohort analysis demonstrates a workflow, not fairness certification.</li><li>Production use would require representative data governance, drift monitoring, model registry, rollback, and review of policy costs.</li><li>The report is reproducible with <code>python scripts/generate_evaluation_report.py --seed {config.seed} --rows {config.rows}</code>.</li></ul></section>
</body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    return metrics