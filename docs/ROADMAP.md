# Roadmap

Each version builds on the previous one. Tag a release (`v1.0`, `v2.0`, ...) when a stage is done.

## V1 — Baseline model (Pandas + scikit-learn) ✅ in progress
- [x] Synthetic credit dataset with controllable drift (`income_multiplier`)
- [x] Schema & range validation
- [x] Feature engineering inside an sklearn `Pipeline` (no train/serve skew)
- [x] Logistic Regression vs Random Forest, credit metrics (ROC AUC, Gini, KS, PR AUC, Brier)
- [x] Best model + `metrics.json` saved to `artifacts/`
- [ ] EDA notebook (`notebooks/01_eda.ipynb`)
- [ ] Probability calibration (`CalibratedClassifierCV`) and threshold selection by business cost

## V2 — Gradient boosting comparison
- [ ] Add XGBoost and LightGBM candidates
- [ ] Stratified K-fold cross-validation
- [ ] Hyperparameter search (Optuna)
- [ ] SHAP-based explainability report

## V3 — MLflow
- [ ] `docker-compose` with MLflow tracking server (PostgreSQL backend + artifact store)
- [ ] Log params, metrics, artifacts, and the model signature for every run
- [ ] Register best model in Model Registry; use aliases (`champion` / `challenger`)

## V4 — Airflow orchestration
- [ ] Load training data into PostgreSQL
- [ ] DAG: `extract_validate → feature_engineering → train → evaluate → register`
- [ ] Evaluation gate: only register if metrics beat thresholds

## V5 — Serving + CI/CD
- [ ] FastAPI inference service (`/predict`, `/health`, `/model-info`) loading the `champion` model
- [ ] Pydantic request/response schemas, input validation
- [ ] Dockerfile + docker-compose
- [ ] GitHub Actions: lint, test, build & push image

## V6 — Monitoring & safe retraining
- [ ] Log every prediction (inputs, score, model version) to PostgreSQL
- [ ] Evidently reports: data drift, prediction drift, data quality
- [ ] Simulate income distribution shift with `income_multiplier`
- [ ] Drift → trigger retraining **candidate**, never auto-promote
- [ ] Champion/challenger evaluation on fresh labelled data + manual approval before promotion

## V7 — Scale (optional)
- [ ] PySpark feature pipeline for large data
- [ ] Kubernetes deployment (kind/minikube), Helm chart
