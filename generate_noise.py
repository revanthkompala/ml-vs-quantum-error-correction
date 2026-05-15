import random
import pandas as pd

# constnatns
NUM_SAMPLES = 8000   # how much data to generate
NUM_QUBITS = 5       # q0 to q4


# noise free data
def create_state(label):
    # repetition encoding: all 1s or all 0s
    if label == 1:
        return [1, 1, 1, 1, 1]
    else:
        return [0, 0, 0, 0, 0]


# addingn noise on quibits
def add_noise(bits):
    noisy = bits.copy()

    # different noise level for each qubit
    noise_probs = [0.10, 0.35, 0.20, 0.35, 0.10]

    for i in range(NUM_QUBITS):

        # random bit flip 
        if random.random() < noise_probs[i]:
            noisy[i] = 1 - noisy[i]

    # correlated noise (neighbor qubits affect each other)
    for i in range(NUM_QUBITS - 1):
        if random.random() < 0.25:
            noisy[i + 1] = noisy[i]

    # random burst error (rare big disturbance)
    if random.random() < 0.05:
        idx = random.randint(0, 4)
        noisy[idx] = random.choice([0, 1])

    return noisy


# full data set generation
data = []

for _ in range(NUM_SAMPLES):

    # random label 
    label = random.randint(0, 1)

    # create clean quantum state
    clean = create_state(label)

    # add noise
    noisy = add_noise(clean)

   
    data.append(noisy + [label])


# save :)
df = pd.DataFrame(data, columns=["q0", "q1", "q2", "q3", "q4", "label"])
df.to_csv("quantum_noise_data.csv", index=False)

print("data is safe")
print(df.head())
