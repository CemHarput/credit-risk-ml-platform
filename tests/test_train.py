import json

import numpy as np
import yaml

from credit_risk.data import generate_credit_data
from credit_risk.predict import score
from credit_risk.train import run


def test_training_end_to_end(tmp_path):
    cfg = {
        "data": {"path": str(tmp_path / "credit.csv"), "n_samples": 3000, "seed": 0},
        "split": {"test_size": 0.25, "random_state": 0},
        "models": {
            "logistic_regression": {"C": 1.0, "max_iter": 500},
            "random_forest": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 20},
        },
        "selection_metric": "roc_auc",
        "decision_threshold": 0.5,
        "output_dir": str(tmp_path / "artifacts"),
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    report = run(config_path)

    assert report["best_model"] in {"logistic_regression", "random_forest"}
    assert report["metrics"][report["best_model"]]["roc_auc"] > 0.7
    saved = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    assert saved["best_model"] == report["best_model"]

    new_customers = generate_credit_data(n_samples=100, seed=99).drop(columns=["default"])
    scores = score(new_customers, tmp_path / "artifacts" / "model.joblib")
    assert len(scores) == 100
    assert np.all((scores["default_probability"] >= 0) & (scores["default_probability"] <= 1))
