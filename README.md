# Credit Risk ML Platform

An end-to-end ML platform that predicts the probability that a customer will default on a loan —
from training and experiment tracking to orchestration, serving, monitoring and **safe** retraining.

## Target architecture

```
PostgreSQL (training data)
        │
Apache Airflow ── extract & validate → feature engineering → train → evaluate
        │
MLflow ── experiments, metrics, artifacts, model registry (champion / challenger)
        │
FastAPI + Docker ── credit risk inference service
        │
Evidently ── data drift, prediction drift, data quality
        │
Drift detected → retrain candidate → evaluate on fresh labels → approval → new model version
```

Development is staged V1 → V7; see [docs/ROADMAP.md](docs/ROADMAP.md).

## Current status: V1

- Synthetic credit portfolio (age, income, employment, loan amount/term, credit history,
  delinquencies, debt-to-income, home ownership, loan purpose) with a known risk function,
  so drift scenarios (e.g. a drop in customer income) can be simulated later.
- Data validation (schema, ranges, missing rates, label sanity).
- Feature engineering and preprocessing inside one sklearn `Pipeline`.
- Logistic Regression vs Random Forest, evaluated with credit-risk metrics:
  ROC AUC, Gini, KS, PR AUC, Brier score, precision/recall/F1.

## Quickstart

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"

python -m credit_risk.data --out data/raw/credit.csv      # generate data
python -m credit_risk.train --config configs/train.yaml   # train & compare models
python -m credit_risk.predict --input data/raw/credit.csv # batch scoring

pytest -q
ruff check .
```

Artifacts are written to `artifacts/` (`model.joblib`, `metrics.json`).

## Project layout

```
configs/            training configuration
src/credit_risk/    data generation & validation, features, training, evaluation, scoring
tests/              unit + end-to-end tests
docs/ROADMAP.md     staged plan V1 → V7
```

## Design principles

- **No training/serving skew** — all transforms live in the persisted pipeline.
- **Drift ≠ auto-deploy** — drift only creates a retraining candidate; promotion requires
  evaluation on fresh labelled data and an explicit approval.
- **Reproducible** — seeded data generation and config-driven training.
