"""
Random Forest - Interview Essentials

Core concept: Ensemble of decision trees. Each tree trained on a bootstrap sample of data
and a random subset of features at each split (decorrelation). Final prediction: majority vote (classification)
or average (regression).

Interview tip: "Bagging + feature randomness reduces variance and overfitting. More trees = more stable;
diminishing returns after a point. Out-of-bag samples can be used for validation."
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


def demo_random_forest():
    """
    Demonstrate Random Forest on Breast Cancer dataset (binary: malignant vs benign).
    Interview: Explain bagging (bootstrap aggregating) and random feature subset.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Dataset: Breast Cancer (sklearn) – malignant vs benign")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Feature importance (interview: "How does RF get feature importance?")
    # Typically mean decrease in impurity (Gini) across splits using that feature
    importances = rf.feature_importances_
    
    top_5_idx = np.argsort(importances)[-5:][::-1]
    top_5_dict = {str(feature_names[i]): round(float(importances[i]), 4) for i in top_5_idx}
    print("=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)
    print(f"Features: {len(feature_names)} (e.g. {[str(n) for n in feature_names[:3]]}...)")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Top 5 feature importances:", top_5_dict)
    print("\nInterview recap:")
    print("- Bagging: each tree on bootstrap sample → reduces variance")
    print("- Random features at each split → decorrelates trees")
    print("- No scaling needed; robust to outliers")
    print("=" * 50)
    
    return rf, importances


if __name__ == "__main__":
    demo_random_forest()
