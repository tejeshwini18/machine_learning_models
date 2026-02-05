# Datasets Used by Module

Each algorithm uses **real datasets** (sklearn or optional GitHub/Kaggle URLs). Some datasets are **reused** across modules where it makes sense (e.g. Iris for decision tree and naive bayes; Wine for logistic and SVM).

| Module | Dataset | Source | Notes |
|--------|---------|--------|------|
| **linear_regression** | California Housing | sklearn `fetch_california_housing` | Regression; median house value. Optional: CSV from GitHub URL. |
| **logistic_regression** | Wine | sklearn `load_wine` | Binary: class 0 vs rest (3 classes binarized). |
| **decision_tree** | Iris | sklearn `load_iris` | 3 classes (setosa, versicolor, virginica). |
| **random_forest** | Breast Cancer | sklearn `load_breast_cancer` | Binary: malignant vs benign; many features. |
| **svm** | Wine | sklearn `load_wine` | 3 classes; same dataset as logistic, multi-class. |
| **naive_bayes** | Iris | sklearn `load_iris` | Same as decision tree – 3 classes. |
| **model_evaluation (metrics)** | Wine | sklearn `load_wine` | Binary class 0 vs rest; Logistic Regression predictions for metrics. |
| **model_evaluation (train_test, CV)** | Iris | sklearn `load_iris` | Same as decision tree/naive_bayes for split and K-fold demos. |
| **model_evaluation (feature_scaling)** | California Housing | sklearn `fetch_california_housing` | Same as linear_regression; features on different scales. |
| **bias_variance_tradeoff** | Synthetic quadratic **or** California Housing | generated / sklearn | Synthetic by default (clear underfit/overfit); optional: MedInc vs value. |
| **outlier_detection** | California Housing **or** synthetic blobs | sklearn / generated | Real data by default; optional synthetic with injected outliers. |

## Summary

- **Regression:** California Housing (linear regression, feature scaling, optional bias-variance and outlier demos).
- **Classification:** Iris (decision tree, naive bayes, train-test/CV); Wine (logistic, SVM, metrics); Breast Cancer (random forest).
- **Reuse:** Same dataset is used in multiple modules where the goal fits (e.g. Iris for tree + NB + CV; Wine for logistic + SVM + metrics).

## Using Your Own Data (GitHub / Kaggle)

- **Linear regression:** Use `load_csv_from_url(url, target_column, feature_columns)` in `linear_regression/linear_regression.py`, or pass a CSV path after downloading from Kaggle.
- **Other modules:** Load your CSV with pandas, extract `X` and `y`, then pass them into the same `fit`/`demo` functions (same interface as sklearn datasets).
