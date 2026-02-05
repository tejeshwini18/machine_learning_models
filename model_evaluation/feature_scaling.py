"""
Feature Scaling & Normalization

Why? Many algorithms (SVM, KNN, gradient descent) are sensitive to feature scale.
Standardization: (x - mean) / std → mean 0, std 1. MinMax: (x - min) / (max - min) → [0,1].

Dataset: California Housing (sklearn) – real features on different scales; ideal for scaling demo.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import fetch_california_housing


def standardize_manual(X: np.ndarray) -> np.ndarray:
    """Z-score: (X - mean) / std. Use when features are roughly normal."""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)


def minmax_manual(X: np.ndarray, feature_range=(0, 1)) -> np.ndarray:
    """MinMax: scale to [a, b]. Sensitive to outliers (min/max)."""
    a, b = feature_range
    min_ = X.min(axis=0)
    max_ = X.max(axis=0)
    return a + (X - min_) * (b - a) / (max_ - min_ + 1e-8)


def demo_feature_scaling():
    """Compare raw vs standardized vs minmax on California Housing (features on different scales)."""
    data = fetch_california_housing()
    X = data.data
    feature_names = data.feature_names
    print("Dataset: California Housing (sklearn) – real census features on different scales")
    print("=" * 50)
    print("FEATURE SCALING & NORMALIZATION")
    print("=" * 50)
    print("Original - column means:", [round(float(x), 4) for x in X.mean(axis=0)])
    print("Original - column stds: ", [round(float(x), 4) for x in X.std(axis=0)])
    
    X_std = StandardScaler().fit_transform(X)
    means_std = [0.0 if abs(x) < 1e-10 else round(float(x), 6) for x in X_std.mean(axis=0)]
    stds_std = [round(float(x), 6) for x in X_std.std(axis=0)]
    print("\nAfter StandardScaler - means:", means_std)
    print("After StandardScaler - stds: ", stds_std)
    
    X_mm = MinMaxScaler().fit_transform(X)
    print("\nAfter MinMaxScaler - min/max (per column): [0, 1]")
    
    # RobustScaler: uses median and IQR - good when outliers present
    X_robust = RobustScaler().fit_transform(X)
    print("After RobustScaler - means ≈ 0, stds ≈ 1 (median/IQR-based; robust to outliers)")
    
    print("\nInterview: StandardScaler for most ML; MinMax for bounded [0,1]; Robust for outliers.")
    print("=" * 50)


if __name__ == "__main__":
    demo_feature_scaling()
