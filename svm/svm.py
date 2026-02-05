"""
Support Vector Machine (SVM) - Interview Essentials

Core concept: Find the hyperplane that maximizes the margin between classes.
For non-linear boundaries: kernel trick (RBF, polynomial) maps data to higher dimensions.

Interview tip: "Hard margin: no misclassification. Soft margin (C): allows slack for noise.
Kernel = inner product in transformed space; RBF is most common. Scale features for SVM!"
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


def demo_svm():
    """
    SVM with linear and RBF kernels on Wine dataset (3 classes).
    Always scale features for distance-based methods.
    """
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Dataset: Wine (sklearn) – 3 wine classes")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Linear SVM (C = regularization; smaller C = wider margin, more margin violations)
    clf_linear = SVC(kernel="linear", C=1.0, random_state=42)
    clf_linear.fit(X_train_s, y_train)
    acc_linear = accuracy_score(y_test, clf_linear.predict(X_test_s))
    
    # RBF kernel (gamma: high = complex boundary, low = smoother)
    clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf_rbf.fit(X_train_s, y_train)
    acc_rbf = accuracy_score(y_test, clf_rbf.predict(X_test_s))
    
    print("=" * 50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 50)
    print(f"Linear SVM accuracy: {acc_linear:.4f}")
    print(f"RBF SVM accuracy: {acc_rbf:.4f}")
    print("\nInterview points:")
    print("- Margin = 2/||w||; maximize margin = minimize ||w||²")
    print("- Support vectors: points on or inside margin")
    print("- Kernel trick: compute in original space without explicit transform")
    print("=" * 50)
    
    return clf_linear, clf_rbf


if __name__ == "__main__":
    demo_svm()
