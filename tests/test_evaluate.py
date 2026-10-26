import numpy as np
import pytest

from credit_risk.evaluate import evaluate, ks_statistic


def test_ks_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, s) == pytest.approx(1.0)


def test_ks_no_separation():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y, s) == pytest.approx(0.0)


def test_evaluate_returns_consistent_gini():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = evaluate(y, s)
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert 0 <= m["brier"] <= 1
