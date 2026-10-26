"""Synthetic credit dataset generation and schema validation.

The generator is deliberately parameterised (e.g. ``income_multiplier``) so that
later versions (V6) can simulate production drift such as a shift in the
customer income distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "credit_history_months",
    "num_open_accounts",
    "num_delinquencies_2y",
    "debt_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE"]
LOAN_PURPOSE = ["debt_consolidation", "home_improvement", "car", "education", "business"]

# Hard domain constraints checked by `validate`.
RANGES = {
    "age": (18, 100),
    "annual_income": (0, None),
    "employment_years": (0, 60),
    "loan_amount": (0, None),
    "loan_term_months": (6, 120),
    "credit_history_months": (0, 1000),
    "num_open_accounts": (0, 100),
    "num_delinquencies_2y": (0, 50),
    "debt_to_income": (0, 5),
}


def generate_credit_data(
    n_samples: int = 20_000,
    seed: int = 42,
    income_multiplier: float = 1.0,
    missing_rate: float = 0.03,
) -> pd.DataFrame:
    """Generate a realistic-looking synthetic credit portfolio with a default label."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_samples)
    employment_years = np.clip(rng.gamma(2.0, 3.0, n_samples), 0, age - 18).round(1)
    annual_income = (
        rng.lognormal(mean=11.0, sigma=0.5, size=n_samples)
        * (1 + 0.02 * employment_years)
        * income_multiplier
    ).round(0)
    loan_amount = (annual_income * rng.uniform(0.05, 0.8, n_samples)).round(-2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.35, 0.15, 0.2])
    credit_history_months = np.clip(
        (age - 18) * 12 * rng.uniform(0.2, 1.0, n_samples), 0, None
    ).astype(int)
    num_open_accounts = rng.poisson(4, n_samples)
    num_delinquencies_2y = rng.poisson(0.3, n_samples)
    debt_to_income = np.clip(rng.beta(2, 5, n_samples) * 1.2, 0, 5).round(3)
    home_ownership = rng.choice(HOME_OWNERSHIP, n_samples, p=[0.4, 0.2, 0.4])
    loan_purpose = rng.choice(LOAN_PURPOSE, n_samples, p=[0.4, 0.15, 0.2, 0.1, 0.15])

    # Ground-truth risk: a logistic function of standardised drivers.
    # Uses absolute income so that an income shift changes the true default rate.
    log_income = np.log(annual_income)
    logit = (
        -2.2
        + 2.5 * (debt_to_income - 0.34)
        - 0.9 * (log_income - 11.0)
        + 1.2 * (loan_amount / annual_income - 0.42)
        + 0.55 * num_delinquencies_2y
        - 0.004 * (credit_history_months - 150)
        - 0.03 * (employment_years - 6)
        + 0.01 * (loan_term_months - 36)
        + 0.06 * (num_open_accounts - 4)
        + np.where(home_ownership == "RENT", 0.3, 0.0)
        + np.where(loan_purpose == "business", 0.45, 0.0)
        + np.where(loan_purpose == "education", 0.2, 0.0)
        + rng.normal(0, 0.3, n_samples)
    )
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            ID_COLUMN: np.arange(1, n_samples + 1),
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "credit_history_months": credit_history_months,
            "num_open_accounts": num_open_accounts,
            "num_delinquencies_2y": num_delinquencies_2y,
            "debt_to_income": debt_to_income,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            TARGET: default,
        }
    )

    # Real data is never complete: knock out a few values to exercise imputation.
    for col in ["annual_income", "employment_years"]:
        mask = rng.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """Return a list of human-readable data quality errors (empty list == valid)."""
    errors: list[str] = []
    required = FEATURES + ([TARGET] if require_target else [])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return [f"missing columns: {missing_cols}"]

    if df.empty:
        errors.append("dataset is empty")

    for col, (lo, hi) in RANGES.items():
        values = df[col].dropna()
        if lo is not None and (values < lo).any():
            errors.append(f"{col}: {(values < lo).sum()} values below {lo}")
        if hi is not None and (values > hi).any():
            errors.append(f"{col}: {(values > hi).sum()} values above {hi}")

    for col in FEATURES:
        rate = df[col].isna().mean()
        if rate > 0.2:
            errors.append(f"{col}: missing rate {rate:.1%} exceeds 20%")

    if require_target:
        if df[TARGET].isna().any():
            errors.append(f"{TARGET}: contains missing labels")
        elif not set(df[TARGET].unique()) <= {0, 1}:
            errors.append(f"{TARGET}: must be binary 0/1")

    return errors


def load_or_generate(path: str | Path, n_samples: int, seed: int) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_credit_data(n_samples=n_samples, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit data")
    parser.add_argument("--out", default="data/raw/credit.csv")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--income-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    df = generate_credit_data(args.n_samples, args.seed, args.income_multiplier)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} (default rate {df[TARGET].mean():.2%})")


if __name__ == "__main__":
    main()
