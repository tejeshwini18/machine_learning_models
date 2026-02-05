"""
Model Evaluation - Classification Metrics

Covers: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC.
Interview: "When to use what? Imbalanced data → Precision/Recall/F1. Probability ranking → ROC-AUC."

Dataset: Wine (sklearn) – we train a classifier to get y_pred/y_proba, then show all metrics.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix: rows = true class, cols = predicted.
    [[TN, FP],
     [FN, TP]]  for binary 0/1.
    Interview: "TN, FP, FN, TP - always define positive class (e.g. 1 = positive)."
    """
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP + TN) / Total. Misleading when classes are imbalanced."""
    return np.mean(y_true == y_pred)


def precision_manual(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Precision = TP / (TP + FP). Of all predicted positive, how many are correct?"""
    pred_pos = y_pred == pos_label
    if pred_pos.sum() == 0:
        return 0.0
    return np.sum((y_true == pos_label) & pred_pos) / np.sum(pred_pos)


def recall_manual(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Recall = TP / (TP + FN). Of all actual positive, how many did we catch?"""
    actual_pos = y_true == pos_label
    if actual_pos.sum() == 0:
        return 0.0
    return np.sum((y_pred == pos_label) & actual_pos) / np.sum(actual_pos)


def f1_manual(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall). Harmonic mean."""
    p = precision_manual(y_true, y_pred, pos_label)
    r = recall_manual(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def demo_classification_metrics(use_real_data: bool = True):
    """
    Demonstrate all classification metrics.
    If use_real_data: train Logistic Regression on Wine (binary: class 0 vs rest) and show metrics.
    Else: use small hand-picked y_true, y_pred for teaching.
    """
    if use_real_data:
        data = load_wine()
        X, y = data.data, data.target
        y_binary = (y == 0).astype(int)  # class 0 vs rest
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = LogisticRegression(max_iter=500).fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        y_true = y_test
        print("Dataset: Wine (sklearn) – binary class 0 vs rest; metrics on test set")
    else:
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
        y_proba = None

    cm = confusion_matrix_manual(y_true, y_pred)
    n_classes = cm.shape[0]
    print("=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nAccuracy:  {accuracy_manual(y_true, y_pred):.4f}")
    if n_classes == 2:
        print(f"Precision: {precision_manual(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_manual(y_true, y_pred):.4f}")
        print(f"F1-Score:  {f1_manual(y_true, y_pred):.4f}")
        if y_proba is not None:
            print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print("\nInterview: Precision vs Recall trade-off. F1 balances both.")
    print("=" * 50)


def demo_roc_auc(y_true: np.ndarray, y_proba: np.ndarray):
    """
    ROC-AUC: Area under ROC curve (TPR vs FPR at various thresholds).
    Interview: "AUC = probability that model ranks a random positive above a random negative."
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    print("ROC-AUC:", round(auc, 4))
    return fpr, tpr, auc


if __name__ == "__main__":
    demo_classification_metrics()
