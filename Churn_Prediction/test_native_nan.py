import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score

train = pd.read_csv("training_set.csv")
y = train["Churned"]
X = train.drop(columns=["Churned"])

# Identify categorical columns
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

# Convert categorical columns to strings, but keep NaNs or fill them with special string?
# CatBoost requires categorical features to be strings/int. If they are floats (which NaNs are), it can complain.
# Let's convert categorical columns to string but keep missing values as 'Unknown' (or None)
for col in cat_cols:
    X[col] = X[col].astype(str)

def add_baseline_features(df):
    df = df.copy()
    df["engagement_ratio"] = (df["TasksCompleted"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))
    df["abandonment_rate"] = (df["TasksAbandoned"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))
    df["learning_intensity"] = (df["TasksCompleted"]/(df["LoggedInDays"]+1))
    df["social_score"] = (df["FriendsCount"] + 10*df["ReferredFriend"])
    return df

X_feat = add_baseline_features(X)

# Here we don't impute the numerical missing values (PrevMonths, PeakActivityHour). We keep them as NaN!
# Let's check how CatBoost performs.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y)):
    X_train_f, X_val_f = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
    y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_state=42,
        nan_mode='Min'  # or 'Max' or 'Forbidden'
    )
    
    model.fit(X_train_f, y_train_f, cat_features=cat_cols)
    probs = model.predict_proba(X_val_f)[:, 1]
    oof_probs[val_idx] = probs

cv_auc = roc_auc_score(y, oof_probs)
cv_logloss = log_loss(y, oof_probs)
print(f"Native NaN (nan_mode='Min') ROC-AUC: {cv_auc:.5f}, Log Loss: {cv_logloss:.5f}")
