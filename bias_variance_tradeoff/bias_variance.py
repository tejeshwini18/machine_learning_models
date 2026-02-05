"""
Bias vs Variance & Underfitting vs Overfitting

Bias: Error from wrong assumptions (e.g. fitting linear when true is non-linear) → underfitting.
Variance: Error from sensitivity to training data (complex model fits noise) → overfitting.
Total Error ≈ Bias² + Variance + Irreducible Error.

Interview: "High bias = underfit (add features, complex model). High variance = overfit (regularization, more data)."

Datasets: (1) Synthetic quadratic for clear underfit/overfit demo. (2) California Housing
(one feature) to show same concepts on real data.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing


def generate_nonlinear_data(n=100, noise=0.5, random_state=42):
    """True function: quadratic. We'll fit linear (high bias) vs polynomial (risk high variance)."""
    np.random.seed(random_state)
    X = np.sort(np.random.uniform(-2, 2, n))
    y = 0.5 * X**2 + X + 1 + np.random.normal(0, noise, n)
    return X.reshape(-1, 1), y


def load_california_single_feature():
    """California Housing: use MedInc (median income) vs target for polynomial demo on real data."""
    data = fetch_california_housing()
    X = data.data[:, :1]  # MedInc only
    y = data.target
    return X, y


def demo_underfitting_overfitting(use_real_data: bool = False):
    """
    Compare: Linear (underfit), Degree-2 (good), Degree-15 (overfit).
    use_real_data: if True, use California Housing (MedInc vs value); else synthetic quadratic.
    """
    if use_real_data:
        X, y = load_california_single_feature()
        print("Dataset: California Housing (sklearn) – MedInc vs median house value")
    else:
        X, y = generate_nonlinear_data(80)
        print("Dataset: Synthetic quadratic (clear underfit/overfit illustration)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = []
    for degree in [1, 2, 15]:
        poly = PolynomialFeatures(degree=degree)
        X_train_p = poly.fit_transform(X_train)
        X_test_p = poly.transform(X_test)
        model = LinearRegression().fit(X_train_p, y_train)
        train_mse = mean_squared_error(y_train, model.predict(X_train_p))
        test_mse = mean_squared_error(y_test, model.predict(X_test_p))
        results.append((degree, train_mse, test_mse))
    
    print("=" * 50)
    print("BIAS vs VARIANCE / UNDERFITTING vs OVERFITTING")
    print("=" * 50)
    print("Degree | Train MSE | Test MSE  | Comment")
    for d, tr, te in results:
        comment = "Underfit (high bias)" if d == 1 else "Overfit (high variance)" if d == 15 else "Good fit"
        print(f"  {d:2d}   |  {tr:.4f}   |  {te:.4f}   | {comment}")
    print("\nInterview: Gap between train and test = overfitting. Similar high error = underfitting.")
    print("=" * 50)
    return results


def demo_regularization():
    """Ridge (L2) reduces variance by penalizing large weights. Helps when overfitting."""
    X, y = generate_nonlinear_data(50, noise=0.3)
    poly = PolynomialFeatures(degree=10)  # High degree → overfit without regularization
    X_p = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_p, y, test_size=0.3, random_state=42)
    
    # No regularization
    lr = LinearRegression().fit(X_train, y_train)
    # With Ridge (alpha = regularization strength)
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    
    lr_mse = mean_squared_error(y_test, lr.predict(X_test))
    ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))
    print("High-degree polynomial (degree=10):")
    print("LinearRegression Test MSE:", f"{lr_mse:.4f}")
    print("Ridge(alpha=1) Test MSE:  ", f"{ridge_mse:.4f}")
    print("Interview: Regularization adds penalty (e.g. λ||w||²) → smaller weights → less overfitting.")
    return lr, ridge


if __name__ == "__main__":
    demo_underfitting_overfitting()
    demo_regularization()
