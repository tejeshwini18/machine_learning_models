"""
Train-Test Split & Cross-Validation

Why split? To estimate generalization error; training on all data would overfit.
Why CV? Single split is noisy; K-fold gives more stable estimate. StratifiedKFold for classification.

Dataset: Iris (sklearn) – classification; same dataset reused for split and CV demos.
"""

import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def load_iris_data():
    """Iris dataset (sklearn) – 3 classes, 4 features. Used for split and CV demos."""
    data = load_iris()
    return data.data, data.target


def demo_train_test_split():
    """Standard 80-20 split on Iris. Always use random_state; stratify for classification."""
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Dataset: Iris (sklearn)")
    print("Train size:", len(X_train), "Test size:", len(X_test))
    return X_train, X_test, y_train, y_test


def demo_kfold():
    """K-Fold: split data into K folds; each fold once as validation, rest as train."""
    X, y = load_iris_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        scores.append(acc)
    print("K-Fold (5) accuracies:", [round(s, 4) for s in scores])
    print("Mean CV accuracy:", round(np.mean(scores), 4))
    return scores


def demo_stratified_kfold():
    """Stratified K-Fold: preserves class proportion in each fold. Use for classification."""
    X, y = load_iris_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=skf, scoring="accuracy")
    print("Stratified K-Fold scores:", scores)
    print("Mean:", scores.mean())
    return scores


def demo_cross_validation():
    """Single call for cross_val_score on Iris - interview: 'I use 5-fold CV to tune hyperparameters.'"""
    X, y = load_iris_data()
    print("Dataset: Iris (sklearn)")
    model = LogisticRegression(max_iter=500)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print("=" * 50)
    print("TRAIN-TEST SPLIT & CROSS-VALIDATION")
    print("=" * 50)
    scores_rounded = [round(float(s), 4) for s in scores]
    print("cross_val_score (5-fold):", scores_rounded)
    print("Mean:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))
    print("\nInterview: Hold-out for final eval; CV for model selection/hyperparameter tuning.")
    print("=" * 50)
    return scores


if __name__ == "__main__":
    demo_cross_validation()
