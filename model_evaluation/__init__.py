"""Model evaluation: metrics, train-test split, cross-validation, feature scaling."""

from .metrics import (
    confusion_matrix_manual,
    accuracy_manual,
    precision_manual,
    recall_manual,
    f1_manual,
    demo_classification_metrics,
    demo_roc_auc,
)
from .train_test_cross_validation import (
    load_iris_data,
    demo_train_test_split,
    demo_kfold,
    demo_stratified_kfold,
    demo_cross_validation,
)
from .feature_scaling import standardize_manual, minmax_manual, demo_feature_scaling

__all__ = [
    "confusion_matrix_manual",
    "accuracy_manual",
    "precision_manual",
    "recall_manual",
    "f1_manual",
    "demo_classification_metrics",
    "demo_roc_auc",
    "load_iris_data",
    "demo_train_test_split",
    "demo_kfold",
    "demo_stratified_kfold",
    "demo_cross_validation",
    "standardize_manual",
    "minmax_manual",
    "demo_feature_scaling",
]
