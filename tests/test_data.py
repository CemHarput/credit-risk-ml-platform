import numpy as np

from credit_risk.data import FEATURES, TARGET, generate_credit_data, validate


def test_generated_data_is_valid():
    df = generate_credit_data(n_samples=2000, seed=1)
    assert validate(df) == []
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert 0.05 < df[TARGET].mean() < 0.4


def test_generation_is_deterministic():
    a = generate_credit_data(n_samples=500, seed=7)
    b = generate_credit_data(n_samples=500, seed=7)
    assert a.equals(b)


def test_income_multiplier_shifts_income_distribution():
    base = generate_credit_data(n_samples=5000, seed=3)
    shifted = generate_credit_data(n_samples=5000, seed=3, income_multiplier=0.7)
    ratio = shifted["annual_income"].median() / base["annual_income"].median()
    assert np.isclose(ratio, 0.7, atol=0.05)


def test_validate_catches_bad_rows():
    df = generate_credit_data(n_samples=200, seed=1)
    df.loc[0, "age"] = 5
    df.loc[1, TARGET] = 3
    errors = validate(df)
    assert any("age" in e for e in errors)
    assert any(TARGET in e for e in errors)


def test_validate_reports_missing_columns():
    df = generate_credit_data(n_samples=50, seed=1).drop(columns=["loan_amount"])
    assert validate(df) == ["missing columns: ['loan_amount']"]
