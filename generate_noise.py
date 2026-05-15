import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("quantum_noise_strong.csv")

print("\nSample data:")
print(df.head())


# =========================================================
# SPLIT FEATURES / LABELS
# =========================================================
X = df.drop("label", axis=1)
y = df["label"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# MODEL (ML DECODER)
# =========================================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# PREDICTION
# =========================================================
y_pred = model.predict(X_test)


# =========================================================
# RESULTS
# =========================================================
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print("ML DECODER RESULTS (UPDATED)")
print("==============================")
print(f"Accuracy: {acc:.4f}\n")

print(classification_report(y_test, y_pred))


# =========================================================
# FEATURE IMPORTANCE (IMPORTANT FOR SCIENCE FAIR)
# =========================================================
importances = model.feature_importances_

print("\nFeature Importance:")
for i, v in enumerate(importances):
    print(f"q{i}: {v:.4f}")