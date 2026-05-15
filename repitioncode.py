import pandas as pd
import numpy as np

# load
df = pd.read_csv("quantum_noise_strong.csv")

X = df.drop("label", axis=1).values
y = df["label"].values


# rep. decder
def repetition_decode(sample):
    ones = np.sum(sample)
    zeros = len(sample) - ones
    return 1 if ones > zeros else 0


# predict
predictions = []

for sample in X:
    pred = repetition_decode(sample)
    predictions.append(pred)

predictions = np.array(predictions)



# accuracy
accuracy = np.mean(predictions == y)

print("REPETITION DECODER RESULTS")
print(f"Accuracy: {accuracy:.4f}")
