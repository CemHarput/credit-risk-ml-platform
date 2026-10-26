"""Batch scoring with a trained model.

Usage:
    python -m credit_risk.predict --input data/raw/credit.csv --out data/processed/scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.data import FEATURES, ID_COLUMN, validate


def score(df: pd.DataFrame, model_path: str | Path = "artifacts/model.joblib") -> pd.DataFrame:
    errors = validate(df, require_target=False)
    if errors:
        raise ValueError("input validation failed:\n  " + "\n  ".join(errors))
    model = joblib.load(model_path)
    out = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        out[ID_COLUMN] = df[ID_COLUMN]
    out["default_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--out", default="data/processed/scores.csv")
    args = parser.parse_args()

    scores = score(pd.read_csv(args.input), args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(f"scored {len(scores)} rows -> {args.out}")


if __name__ == "__main__":
    main()
