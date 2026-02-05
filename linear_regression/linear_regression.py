"""
Linear Regression - Interview Essentials

Core concept: Predict a continuous target (y) using a linear combination of features (X).
Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Interview tip: "We minimize the sum of squared residuals (MSE) to find the best-fit line.
Closed-form solution: β = (X'X)⁻¹X'y (Normal Equation). Alternative: Gradient Descent."

Datasets: Uses real data from sklearn (California Housing) or CSV from GitHub/Kaggle URL.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression using the Normal Equation: β = (X'X)⁻¹ X'y
    
    Why Normal Equation? Direct solution, no iteration. Works well when n_features is moderate.
    Add column of 1s for intercept (β₀).
    """
    # Add bias term (column of 1s) for intercept - every interview asks "how do you get β₀?"
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal Equation: β = (X'X)⁻¹ X'y
    # X'X must be invertible; if not (collinearity), use np.linalg.lstsq or regularization
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    coefficients = np.linalg.solve(XtX, Xty)  # More stable than inv() @ Xty
    
    return coefficients


def predict(X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict using y = Xβ (with bias column added inside)."""
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    return X_with_bias @ coefficients


def load_california_housing():
    """
    Load California Housing dataset (real census data from sklearn).
    Target: median house value; features: MedInc, HouseAge, AveRooms, etc.
    """
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names


def load_csv_from_url(url: str, target_column: str, feature_columns: list = None):
    """
    Load a CSV from GitHub raw URL or any public CSV URL (e.g. Kaggle export link).
    Returns X (features), y (target), and feature names.

    Example GitHub raw URL:
        'https://raw.githubusercontent.com/username/repo/branch/path/file.csv'
    For Kaggle: download CSV, host somewhere or use kaggle CLI; or use URL if public.
    """
    df = pd.read_csv(url)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y, feature_columns


def demo_linear_regression(use_dataset: str = "california"):
    """
    Demonstrate linear regression with real datasets.

    use_dataset: 'california' (sklearn California Housing) or 'url' (requires URL).
    For custom CSV from GitHub: use load_csv_from_url() then fit_linear_regression().
    """
    if use_dataset == "california":
        X, y, feature_names = load_california_housing()
        print("Dataset: California Housing (sklearn – real census data)")
    else:
        # Example: use a well-known regression CSV from GitHub
        # Housing dataset (alternative – small, no auth)
        url = (
            "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
            "datasets/housing/housing.csv"
        )
        try:
            df = pd.read_csv(url)
            # Median house value as target; numeric features only
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = "median_house_value"
            if target_col not in numeric:
                numeric.append(target_col)
            feature_cols = [c for c in numeric if c != target_col]
            X = df[feature_cols].fillna(df[feature_cols].median()).values
            y = df[target_col].values
            feature_names = feature_cols
            print("Dataset: Housing (GitHub – handson-ml2)")
        except Exception as e:
            print(f"Could not load URL dataset: {e}. Using California Housing.")
            X, y, feature_names = load_california_housing()
            print("Dataset: California Housing (sklearn)")

    # Best practice: scale features (interview: "Why scale?")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Our implementation
    coef = fit_linear_regression(X_train, y_train)
    y_pred_custom = predict(X_test, coef)

    mse = mean_squared_error(y_test, y_pred_custom)
    r2 = r2_score(y_test, y_pred_custom)

    print("=" * 50)
    print("LINEAR REGRESSION - Custom Implementation")
    print("=" * 50)
    print(f"Features: {list(feature_names)}")
    print(f"Intercept (β₀): {coef[0]:.4f}")
    coef_dict = {name: round(float(c), 4) for name, c in zip(feature_names, coef[1:])}
    print(f"Coefficients: {coef_dict}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print("\nInterview recap: R² = 1 - (SS_res/SS_tot); MSE = mean of (y - ŷ)²")
    print("=" * 50)

    return coef, X_test, y_test, y_pred_custom


if __name__ == "__main__":
    # Use real dataset: California Housing (default)
    demo_linear_regression(use_dataset="california")
