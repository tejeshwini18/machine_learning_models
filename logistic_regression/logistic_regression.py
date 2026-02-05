"""
Logistic Regression - Interview Essentials

Core concept: Binary classification using the logistic (sigmoid) function.
P(y=1|X) = 1 / (1 + e^(-z)) where z = β₀ + β'X

Interview tip: "It's a linear model for classification. We use log-loss (cross-entropy)
and fit with gradient descent or IRLS. Output is probability; we threshold at 0.5 for class."
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import load_wine


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid σ(z) = 1/(1+e^(-z)). Maps real numbers to (0, 1) for probability."""
    # Clip to avoid overflow in exp(-z) when z is very negative
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
) -> np.ndarray:
    """
    Fit logistic regression using Gradient Descent.
    
    Loss: Binary Cross-Entropy = -[y*log(p) + (1-y)*log(1-p)]
    Gradient: ∂L/∂β = X'(p - y)  (same form as linear regression but p = sigmoid(Xβ))
    """
    n_samples, n_features = X.shape
    # Add bias column
    X_bias = np.column_stack([np.ones(n_samples), X])
    beta = np.zeros(n_features + 1)
    
    for _ in range(n_iterations):
        # Forward: probability predictions
        z = X_bias @ beta
        p = sigmoid(z)
        # Gradient of cross-entropy w.r.t. beta
        gradient = X_bias.T @ (p - y) / n_samples
        beta -= learning_rate * gradient
    
    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return P(y=1|X)."""
    X_bias = np.column_stack([np.ones(len(X)), X])
    return sigmoid(X_bias @ beta)


def predict(X: np.ndarray, beta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Class prediction: 1 if P(y=1) >= threshold, else 0."""
    return (predict_proba(X, beta) >= threshold).astype(int)


def load_wine_binary():
    """Wine dataset (sklearn): 3 classes. We binarize to class 0 vs rest for logistic regression."""
    data = load_wine()
    X, y = data.data, data.target
    y_binary = (y == 0).astype(int)  # class 0 vs rest
    return X, y_binary, data.feature_names


def demo_logistic_regression():
    """Demonstrate logistic regression with Wine dataset (binary: class 0 vs rest)."""
    X, y, feature_names = load_wine_binary()
    print("Dataset: Wine (sklearn) – binary: class 0 vs rest")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    beta = fit_logistic_regression(X_train_s, y_train)
    y_pred = predict(X_test_s, beta)
    y_proba = predict_proba(X_test_s, beta)
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    
    coef_dict = {name: round(float(c), 4) for name, c in zip(feature_names, beta[1:])}
    print("=" * 50)
    print("LOGISTIC REGRESSION - Custom Implementation")
    print("=" * 50)
    print(f"Features: {list(feature_names)}")
    print(f"Intercept (β₀): {beta[0]:.4f}")
    print(f"Coefficients: {coef_dict}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print("\nInterview: Why log-loss? It's the MLE for Bernoulli; penalizes wrong confidence.")
    print("=" * 50)
    
    return beta, X_test_s, y_test, y_pred


if __name__ == "__main__":
    demo_logistic_regression()
