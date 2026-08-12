# Failure Modes

| Failure mode | Current behavior | Production extension |
| --- | --- | --- |
| Incomplete evidence | Route to `review` with a stable reason code | Track source-level completeness and alert on drift |
| Low identity confidence | Route to `review` | Add a separately validated identity workflow |
| High predicted risk | Apply explicit policy threshold | Monitor threshold stability and outcome cost |
| Calibration drift | Not detected automatically | Compare rolling reliability curves and Brier score |
| Synthetic data mismatch | Documented as a limitation | Validate against an approved representative dataset |
| Model version mismatch | Version returned in API response | Enforce registry pinning and deployment checks |
| Policy change without review | Not possible through the current API | Store policy versions and require change approval |

The project intentionally fails conservatively when evidence is insufficient. It does not treat model output as the only source of truth.
