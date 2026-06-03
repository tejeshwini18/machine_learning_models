import pandas as pd
from catboost import CatBoostClassifier

train = pd.read_csv("training_set.csv")
test = pd.read_csv("test_set.csv")

print(train.columns)
print(test.columns)

for df in [train,test]:
    df["engagement_ratio"] = (df["TasksCompleted"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))

    df["abandonment_rate"] = (df["TasksAbandoned"]/(df["TasksCompleted"]+df["TasksAbandoned"]+1))

    df["learning_intensity"] = (df["TasksCompleted"]/(df["LoggedInDays"]+1))

    df["social_score"] = (df["FriendsCount"] + 10*df["ReferredFriend"])

y = train["Churned"]
X = train.drop(columns=["Churned"])

# cat_cols = ["Gender","AcquisitionSource","PreferredStudyTime","StartedNewLanguage","ReceivedReminderEmail"]

cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

for col in X.columns:
    if col in cat_cols:
        X[col] = X[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")
    else:
        median = X[col].median()
        X[col] = X[col].fillna(median)
        test[col] = test[col].fillna(median)

model = CatBoostClassifier(iterations=500,depth=6,learning_rate=0.05,loss_function="Logloss",verbose=False,random_state=42)
model.fit(X,y,cat_features=cat_cols)


prob = model.predict_proba(test)[:,1]
pred = (prob >= 0.5).astype(int)

submission = pd.DataFrame({
    "Churning Probability":prob,
    "Churned":pred
})
print(submission)
submission.to_csv("prediction.csv",index=False)
