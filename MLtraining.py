import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# load data from CSV. Please make sure it is correctily loaded
df = pd.read_csv("quantum_noise_strong.csv")  


# labels
X = df.drop("label", axis=1)
y = df["label"]


# training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STRONGER MODEL 
model = GradientBoostingClassifier(
    n_estimators=400,     # more learners
    learning_rate=0.05,   # stable learning
    max_depth=3
)

model.fit(X_train, y_train)


#predictions
y_pred = model.predict(X_test)


#accuracy
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Test Accuracy: {acc:.4f}\n")

print("==============================")

print(classification_report(y_test, y_pred))


# cross-validiation
cv_scores = cross_val_score(model, X, y, cv=5)

print("CROSS-VALIDATION RESULTS")
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))


# FEATURE IMPORTANCE
importances = model.feature_importances_

print("\n==============================")
print("feature importance")
print("==============================")
for i, val in enumerate(importances):
    print(f"q{i}: {val:.4f}")
