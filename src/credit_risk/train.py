"""V1 training entrypoint: train candidate models, compare, persist the best one.

Usage:
    python -m credit_risk.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk import __version__
from credit_risk.data import FEATURES, TARGET, load_or_generate, validate
from credit_risk.evaluate import evaluate
from credit_risk.features import build_preprocessor


def build_models(cfg: dict) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    if "logistic_regression" in cfg:
        models["logistic_regression"] = Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=True)),
                ("clf", LogisticRegression(class_weight="balanced", **cfg["logistic_regression"])),
            ]
        )
    if "random_forest" in cfg:
        models["random_forest"] = Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced", n_jobs=-1, random_state=42, **cfg["random_forest"]
                    ),
                ),
            ]
        )
    return models


def run(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())

    df = load_or_generate(cfg["data"]["path"], cfg["data"]["n_samples"], cfg["data"]["seed"])
    errors = validate(df)
    if errors:
        raise ValueError("data validation failed:\n  " + "\n  ".join(errors))

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y,
        random_state=cfg["split"]["random_state"],
    )

    threshold = cfg["decision_threshold"]
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in build_models(cfg["models"]).items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, scores, threshold)
        fitted[name] = model
        print(f"{name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    metric = cfg["selection_metric"]
    best = max(results, key=lambda n: results[n][metric])
    print(f"\nbest model by {metric}: {best}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "model.joblib")
    report = {
        "best_model": best,
        "selection_metric": metric,
        "decision_threshold": threshold,
        "metrics": results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": float(y.mean()),
        "features": FEATURES,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
