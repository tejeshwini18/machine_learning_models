# Machine Learning Models – Interview-Ready Repository

A hands-on collection of **machine learning concepts and algorithms** implemented in clean, readable Python. Each module focuses on one core idea and is designed to be **easy to explain and reproduce in technical interviews**.

## Folder Structure

```
machine_learning_models/
├── linear_regression/      # Normal equation, MSE, R²
├── logistic_regression/    # Sigmoid, gradient descent, binary classification
├── decision_tree/          # Gini, entropy, information gain
├── random_forest/          # Bagging, feature importance
├── svm/                    # Linear & RBF kernels, margin
├── naive_bayes/            # Gaussian NB, Bayes' theorem
├── model_evaluation/       # Metrics, train-test split, CV, feature scaling
├── bias_variance_tradeoff/ # Underfitting, overfitting, regularization
└── outlier_detection/      # Z-score, IQR, Isolation Forest
```

## Datasets

Each module uses **real datasets** (sklearn or optional GitHub/Kaggle). Different algorithms use different datasets where it makes sense; some reuse the same one (e.g. Iris for decision tree and naive bayes, Wine for logistic and SVM). See **[DATASETS.md](DATASETS.md)** for the full mapping.

| Algorithm / Concept | Dataset |
|--------------------|---------|
| Linear regression | California Housing (sklearn); optional CSV from GitHub URL |
| Logistic regression | Wine (sklearn, binary) |
| Decision tree | Iris (sklearn) |
| Random forest | Breast Cancer (sklearn) |
| SVM | Wine (sklearn, 3 classes) |
| Naive Bayes | Iris (sklearn) |
| Metrics / CV / Scaling | Wine, Iris, California Housing |

## Topics Covered

| Topic | Location | Interview Focus |
|-------|----------|-----------------|
| **Bias vs Variance** | `bias_variance_tradeoff/` | Underfitting vs overfitting, regularization |
| **Train-test split & CV** | `model_evaluation/train_test_cross_validation.py` | K-fold, StratifiedKFold |
| **Confusion matrix** | `model_evaluation/metrics.py` | TN, FP, FN, TP |
| **Accuracy, Precision, Recall, F1** | `model_evaluation/metrics.py` | When to use which |
| **ROC-AUC** | `model_evaluation/metrics.py` | Probability ranking |
| **Feature scaling** | `model_evaluation/feature_scaling.py` | StandardScaler, MinMax, Robust |
| **Outliers** | `outlier_detection/` | Z-score, IQR, winsorize, Isolation Forest |

## Requirements

- Python 3.8+
- NumPy, Pandas, scikit-learn (see `requirements.txt`)

Install:

```bash
pip install -r requirements.txt
```

## How to Run

Each module can be run as a script to see a small demo and printed explanations:

```bash
python -m machine_learning_models.linear_regression.linear_regression
python -m machine_learning_models.logistic_regression.logistic_regression
python -m machine_learning_models.decision_tree.decision_tree
python -m machine_learning_models.random_forest.random_forest
python -m machine_learning_models.svm.svm
python -m machine_learning_models.naive_bayes.naive_bayes
python -m machine_learning_models.model_evaluation.metrics
python -m machine_learning_models.model_evaluation.train_test_cross_validation
python -m machine_learning_models.model_evaluation.feature_scaling
python -m machine_learning_models.bias_variance_tradeoff.bias_variance
python -m machine_learning_models.outlier_detection.outlier_detection
```

Or from the repo root:

```bash
cd machine_learning_models
python linear_regression/linear_regression.py
# etc.
```

## Design Principles

- **One concept per module** – easy to locate and review.
- **Simple implementations** – NumPy/sklearn; no unnecessary complexity.
- **Comments** – explain *what*, *why*, and *interview relevance*.
- **Real data** – sklearn (California Housing, Iris, Wine, Breast Cancer) and optional GitHub/Kaggle CSV URLs.
- **Modular & reusable** – functions you can copy into an interview or project.

## Goal

Build a strong **hands-on** foundation so you can confidently **write and explain** ML code under interview pressure—not just theory, but correct, clean code.
