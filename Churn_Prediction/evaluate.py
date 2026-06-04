import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score, classification_report

print("Loading data...")
train = pd.read_csv("training_set.csv")

# Extract target and features
y = train["Churned"]
X = train.drop(columns=["Churned"])

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

# Baseline preprocessing (imputing values)
def preprocess_baseline(X_train, X_val, cat_cols):
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in X_train.columns:
        if col in cat_cols:
            X_train[col] = X_train[col].fillna("Unknown")
            X_val[col] = X_val[col].fillna("Unknown")
        else:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_val[col] = X_val[col].fillna(median)
    return X_train, X_val

# Baseline features
def add_baseline_features(df):
    df = df.copy()
    df["engagement_ratio"] = (df["TasksCompleted"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))
    df["abandonment_rate"] = (df["TasksAbandoned"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))
    df["learning_intensity"] = (df["TasksCompleted"]/(df["LoggedInDays"]+1))
    df["social_score"] = (df["FriendsCount"] + 10*df["ReferredFriend"])
    return df

print("Adding baseline features...")
X_feat = add_baseline_features(X)

# Define K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
oof_probs = np.zeros(len(train))

print("Running 5-fold cross validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y)):
    print(f"--- Fold {fold+1} ---")
    X_train_f, X_val_f = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
    y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
    
    # Preprocess
    X_train_f, X_val_f = preprocess_baseline(X_train_f, X_val_f, cat_cols)
    
    # Train model
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_state=42
    )
    
    model.fit(X_train_f, y_train_f, cat_features=cat_cols)
    
    # Predict
    probs = model.predict_proba(X_val_f)[:, 1]
    oof_probs[val_idx] = probs
    oof_preds[val_idx] = (probs >= 0.5).astype(int)

# Calculate metrics
cv_auc = roc_auc_score(y, oof_probs)
cv_logloss = log_loss(y, oof_probs)
cv_f1 = f1_score(y, oof_preds)

print("\n=== Baseline Cross-Validation Metrics ===")
print(f"ROC-AUC:  {cv_auc:.5f}")
print(f"Log Loss: {cv_logloss:.5f}")
print(f"F1-Score: {cv_f1:.5f}")

# Find optimal threshold for F1-score
best_thresh = 0.5
best_f1 = cv_f1
for thresh in np.linspace(0.1, 0.9, 81):
    f1 = f1_score(y, (oof_probs >= thresh).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Optimal Threshold: {best_thresh:.2f} (F1: {best_f1:.5f})")
print("\nClassification Report (threshold=0.5):")
print(classification_report(y, oof_preds))
print("\nClassification Report (optimal threshold):")
print(classification_report(y, (oof_probs >= best_thresh).astype(int)))
