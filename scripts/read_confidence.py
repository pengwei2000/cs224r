import matplotlib.pyplot as plt
import numpy as np
confidences = []
ref_confidences = []
with open('../output/confidences_dpo.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            try:
                confidence = float(parts[1])
                confidences.append(confidence)
            except ValueError:
                continue
with open('../output/confidences_ref.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            try:
                ref_confidence = float(parts[1])
                ref_confidences.append(ref_confidence)
            except ValueError:
                continue
# Determine the combined min and max for both lists
all_confidences = confidences + ref_confidences
min_conf = min(all_confidences)
max_conf = max(all_confidences)

# Plot both histograms with the same range and bins
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(confidences, bins=100, range=(min_conf, max_conf), alpha=0.5, label='DPO', edgecolor='black')
ax.hist(ref_confidences, bins=100, range=(min_conf, max_conf), alpha=0.5, label='Ref', edgecolor='black')
ax.legend()
ax.set_xlabel('Rewards')
ax.set_ylabel('Frequency')
ax.set_xlim(-2000, 2000)
ax.set_title('Histogram of Rewards')
fig.savefig('../output/rewards_histogram.png', bbox_inches='tight', dpi=300)
plt.close(fig)