"""Tests for model evaluation metrics with known inputs."""

import numpy as np

from src.models.xgboost_baseline import evaluate


def test_perfect_predictions():
    """Perfect predictions should give MAE=0, RMSE=0, R²=1, Bias=0."""
    y = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    m = evaluate(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["bias"] == 0.0
    assert m["n"] == 5


def test_known_bias():
    """Constant over-prediction should show in bias."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 22.0, 32.0])  # +2m bias
    m = evaluate(y_true, y_pred)
    assert abs(m["bias"] - 2.0) < 1e-10
    assert abs(m["mae"] - 2.0) < 1e-10
    assert abs(m["rmse"] - 2.0) < 1e-10


def test_known_mae():
    """Symmetric errors should give expected MAE."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([14.0, 16.0])  # errors: +4, -4
    m = evaluate(y_true, y_pred)
    assert abs(m["mae"] - 4.0) < 1e-10
    assert abs(m["bias"] - 0.0) < 1e-10  # symmetric → zero bias


def test_r2_constant_prediction():
    """Predicting the mean should give R²=0."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.full(4, 25.0)  # predicting the mean
    m = evaluate(y_true, y_pred)
    assert abs(m["r2"] - 0.0) < 1e-10


def test_n_count():
    """n should match the number of samples."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    m = evaluate(y, y)
    assert m["n"] == 7
