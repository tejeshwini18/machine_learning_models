"""
Decision Tree - Interview Essentials

Core concept: Split data recursively by feature and threshold to maximize information gain
(or minimize impurity). Common criteria: Gini impurity, Entropy.

Interview tip: "We choose the split that maximizes information gain. Tree grows until
stopping condition (max_depth, min_samples_leaf). Pruning reduces overfitting."
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def gini_impurity(y: np.ndarray) -> float:
    """
    Gini impurity: 1 - Σ p_i². Zero when all samples same class.
    Interview: "Gini is faster to compute; Entropy gives similar splits."
    """
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)


def entropy(y: np.ndarray) -> float:
    """Entropy: -Σ p_i * log2(p_i). Measures disorder."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]  # avoid log(0)
    return -np.sum(p * np.log2(p))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, criterion: str = "gini") -> float:
    """
    IG = impurity(parent) - weighted_avg(impurity(left), impurity(right)).
    We want splits that maximize IG.
    """
    imp = gini_impurity if criterion == "gini" else entropy
    n = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)
    return imp(y_parent) - (n_l / n) * imp(y_left) - (n_r / n) * imp(y_right)


def find_best_split(X: np.ndarray, y: np.ndarray, criterion: str = "gini") -> tuple:
    """
    Brute-force: try each feature and (unique values as thresholds).
    Returns (best_feature, best_threshold, best_ig) or (None, None, 0).
    """
    best_ig, best_feat, best_thr = 0.0, None, None
    n_features = X.shape[1]
    
    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            left = y[X[:, feat] <= t]
            right = y[X[:, feat] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            ig = information_gain(y, left, right, criterion)
            if ig > best_ig:
                best_ig, best_feat, best_thr = ig, feat, t
    
    return best_feat, best_thr, best_ig


def demo_decision_tree():
    """Use sklearn Decision Tree on Iris dataset (3 classes: setosa, versicolor, virginica)."""
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Dataset: Iris (sklearn) – 3 flower classes")
    
    # Sklearn tree (clean, interview-ready to explain)
    clf = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("=" * 50)
    print("DECISION TREE")
    print("=" * 50)
    print(f"Features: {list(feature_names)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Gini impurity at root: {gini_impurity(y_train):.4f}")
    print("\nKey interview points:")
    print("- Splits chosen by max Information Gain (Gini or Entropy)")
    print("- max_depth limits overfitting; min_samples_leaf avoids tiny leaves")
    print("- No feature scaling needed (tree splits on thresholds)")
    print("=" * 50)
    
    return clf, X_train, y_train


if __name__ == "__main__":
    demo_decision_tree()
