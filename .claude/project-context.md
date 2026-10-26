# Project context

This file is project-specific guidance, not a Claude Code settings schema.

- Stack: Python 3.12 (`.venv`, created with `py -3.12 -m venv .venv`), pandas,
  scikit-learn, PyYAML, joblib. Package lives in `src/credit_risk/` (setuptools,
  editable install via `pip install -e ".[dev]"`).
- Architecture:
  - `data.py` — synthetic data generation (`income_multiplier` simulates drift) and
    `validate()` schema/range checks. Feature lists (`FEATURES`, `TARGET`) are defined here.
  - `features.py` — feature engineering + preprocessing as one sklearn `Pipeline`.
    All transforms must stay inside the persisted pipeline (no train/serve skew).
  - `evaluate.py` — credit metrics (ROC AUC, Gini, KS, PR AUC, Brier, P/R/F1).
  - `train.py` — config-driven training (`configs/train.yaml`), model comparison,
    writes `artifacts/model.joblib` and `artifacts/metrics.json`.
  - `predict.py` — batch scoring with validation.
- Test/lint commands (run from repo root):
  - `.venv/Scripts/pytest -q`
  - `.venv/Scripts/ruff check .`
  - Smoke training: `.venv/Scripts/python -m credit_risk.train --config configs/train.yaml`
- CI: `.github/workflows/ci.yml` runs `ruff check .` and `pytest -q` on Python 3.12.
- Runtime requirements: no external services yet. MLflow, Airflow, PostgreSQL, FastAPI,
  Docker and Evidently arrive in later stages (see `docs/ROADMAP.md`); report them as
  missing rather than installing silently.
- Domain constraints:
  - Drift detection must never auto-promote a model; promotion requires evaluation on
    fresh labelled data and explicit approval.
  - Keep data generation seeded and training config-driven (reproducibility).
  - `data/`, `artifacts/`, `mlruns/` are generated and git-ignored.
- Additional constraints: follow the staged plan in `docs/ROADMAP.md`; one stage per feature.

Do not invent a command, silently install dependencies, or report unrun checks
as passing. Run focused relevant tests plus the required repository checks.
