import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("quantum_noise_strong.csv")

X = df.drop("label", axis=1).values
y = df["label"].values


# =========================================================
# REPETITION DECODER (MAJORITY VOTE)
# =========================================================
def repetition_decode(sample):
    ones = np.sum(sample)
    zeros = len(sample) - ones
    return 1 if ones > zeros else 0


# =========================================================
# PREDICT
# =========================================================
predictions = []

for sample in X:
    pred = repetition_decode(sample)
    predictions.append(pred)

predictions = np.array(predictions)


# =========================================================
# ACCURACY
# =========================================================
accuracy = np.mean(predictions == y)

print("\n==============================")
print("REPETITION DECODER RESULTS")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")