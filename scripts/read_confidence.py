import matplotlib.pyplot as plt
import numpy as np
confidences = []
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
with open('../output/confidences_ref.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            try:
                confidence = float(parts[1])
                confidences.append(confidence)
            except ValueError:
                continue

plt.hist(confidences, bins=100, edgecolor='black')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.xlim(-2000, 2000)
plt.title('Histogram of Confidences')
plt.show()