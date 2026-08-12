# Architecture Notes

## Responsibility boundaries

The system has four deliberately separate responsibilities:

1. `synthetic.py` creates reproducible, non-production data.
2. `model.py` trains the estimator, calibrates probabilities, and reports holdout metrics.
3. `policy.py` applies explicit operational rules to the model output and evidence quality.
4. `api.py` exposes the contract without exposing training internals.

A policy change must not silently retrain or alter the model. A model change must be evaluated independently before it is used by policy.

## Why the model is calibrated

A ranking model can order observations correctly while producing probabilities that are not meaningful as probabilities. The isotonic calibration stage makes the reported value more useful for threshold analysis, while the untouched final holdout provides a separate estimate of generalization.

## Why incomplete evidence routes to review

The system treats missing or low-quality evidence as a workflow state, not as a favorable model feature. This avoids turning data absence into an implicit approval signal.

## Operational extension points

A production implementation would add:

- a versioned model registry;
- persisted training and evaluation artifacts;
- drift and data-quality monitoring;
- policy configuration with change history;
- access control and audit storage;
- rollback procedures.
