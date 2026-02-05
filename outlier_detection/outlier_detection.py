"""
Outlier Detection & Handling Techniques

Outliers: points that deviate significantly from the rest. Can skew mean, variance, and model fit.
Handling: Remove, cap (winsorize), transform, or use robust methods (median, IQR, RobustScaler).

Dataset: California Housing (sklearn) – real data; some features have heavy tails. Option: synthetic blobs + outliers.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing, make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler


def zscore_outliers(X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Z-score: |z| > threshold → outlier. Assumes roughly normal distribution.
    z = (x - mean) / std. Sensitive to outliers (mean/std themselves get affected).
    """
    z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
    return np.any(z > threshold, axis=1)  # boolean mask: True = outlier


def iqr_outliers(X: np.ndarray, column: int = 0, k: float = 1.5) -> np.ndarray:
    """
    IQR method: outlier if x < Q1 - k*IQR or x > Q3 + k*IQR.
    More robust than z-score (uses quartiles). k=1.5 is common.
    """
    q1 = np.percentile(X[:, column], 25)
    q3 = np.percentile(X[:, column], 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (X[:, column] < lower) | (X[:, column] > upper)


def winsorize(X: np.ndarray, column: int, lower_p: float = 0.05, upper_p: float = 0.95) -> np.ndarray:
    """Cap values at percentiles instead of removing. Keeps sample size."""
    X = X.copy()
    low = np.percentile(X[:, column], lower_p * 100)
    high = np.percentile(X[:, column], upper_p * 100)
    X[:, column] = np.clip(X[:, column], low, high)
    return X


def demo_outlier_detection(use_real_data: bool = True):
    """
    Compare Z-score, IQR, and Isolation Forest.
    use_real_data: True = California Housing (real data, some heavy-tailed features);
                   False = synthetic blobs + injected outliers.
    """
    if use_real_data:
        data = fetch_california_housing()
        X = data.data
        feature_names = data.feature_names
        print("Dataset: California Housing (sklearn) – real data; outliers in feature space")
    else:
        X, _ = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
        X = np.vstack([X, np.array([[10, 10], [-8, -8], [12, -5]])])
        feature_names = ["f0", "f1"]

    n = len(X)
    print("=" * 50)
    print("OUTLIER DETECTION & HANDLING")
    print("=" * 50)
    print(f"Total samples: {n}\n")

    # Z-score (on standardized data for multi-dim)
    X_scaled = StandardScaler().fit_transform(X)
    z_mask = zscore_outliers(X_scaled, threshold=3.0)
    z_count = z_mask.sum()
    print(f"Z-score (|z|>3):        {z_count:4d}  ({100 * z_count / n:.1f}%)")

    # IQR on first column
    iqr_mask = iqr_outliers(X, column=0)
    iqr_count = iqr_mask.sum()
    print(f"IQR (col 0, k=1.5):     {iqr_count:4d}  ({100 * iqr_count / n:.1f}%)")

    # Isolation Forest (sklearn) - tree-based; contamination = expected outlier fraction
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso.fit_predict(X)
    iso_count = (iso_pred == -1).sum()
    print(f"Isolation Forest (5%):  {iso_count:4d}  ({100 * iso_count / n:.1f}%)")

    # Elliptic Envelope (assumes Gaussian; fits robust covariance)
    ell = EllipticEnvelope(contamination=0.05)
    ell_pred = ell.fit_predict(X)
    ell_count = (ell_pred == -1).sum()
    print(f"Elliptic Envelope (5%): {ell_count:4d}  ({100 * ell_count / n:.1f}%)")

    print("\nInterview: Z/IQR = threshold-based; IF/Elliptic use contamination (expected %).")
    print("Remove vs cap (winsorize) vs robust methods. IQR and Isolation Forest are robust.")
    print("=" * 50)
    return X, z_mask, iqr_mask, iso_pred


if __name__ == "__main__":
    demo_outlier_detection()
