"""
Naive Bayes - Interview Essentials

Core concept: Classify using Bayes' theorem with "naive" assumption: features are
conditionally independent given the class. P(Y|X) ∝ P(Y) * Π P(X_i|Y).

Interview tip: "Fast, works well with small data. Naive assumption is often violated
but model is robust. Types: Gaussian (continuous), Multinomial (counts), Bernoulli (binary)."
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def demo_naive_bayes():
    """
    Gaussian Naive Bayes on Iris dataset (3 classes).
    P(X_i|Y) is Gaussian per feature per class.
    """
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Dataset: Iris (sklearn) – 3 flower classes")
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    print("=" * 50)
    print("NAIVE BAYES (Gaussian)")
    print("=" * 50)
    print(f"Features: {list(feature_names)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nInterview: P(Y=k|X) ∝ P(Y=k) * Π P(X_i|Y=k). Gaussian NB assumes P(X_i|Y) is normal.")
    print("No hyperparameters to tune; fast training. Good baseline for text (Multinomial NB).")
    print("=" * 50)
    
    return nb


if __name__ == "__main__":
    demo_naive_bayes()
