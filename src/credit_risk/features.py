"""Feature engineering and preprocessing.

Everything lives inside a single sklearn Pipeline so the exact same transforms
run at training time and at inference time (no training/serving skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ENGINEERED_FEATURES = ["loan_to_income", "monthly_installment_to_income", "log_income"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    income = df["annual_income"].where(df["annual_income"] > 0)
    df["loan_to_income"] = df["loan_amount"] / income
    df["monthly_installment_to_income"] = (df["loan_amount"] / df["loan_term_months"]) / (
        income / 12
    )
    df["log_income"] = np.log1p(income)
    return df


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    columns = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES + ENGINEERED_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered_features)),
            ("columns", columns),
        ]
    )
