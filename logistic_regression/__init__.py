"""Logistic Regression module - binary classification with sigmoid."""

from .logistic_regression import (
    sigmoid,
    fit_logistic_regression,
    predict,
    predict_proba,
    load_wine_binary,
    demo_logistic_regression,
)

__all__ = [
    "sigmoid",
    "fit_logistic_regression",
    "predict",
    "predict_proba",
    "load_wine_binary",
    "demo_logistic_regression",
]
