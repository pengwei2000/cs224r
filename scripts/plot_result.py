import numpy as np
import matplotlib.pyplot as plt

# Original data
models = ["SFT baseline", "DPO", "Minll", "Unll", "Contrastll", "Minll+DPO", "Unll+DPO", "Contrastll+DPO"]
result = [0.500, 0.759, 0.893, 0.898, 0.890, 0.889, 0.977, 0.895]

# Define positions for bars
# We'll make "Minll" and "Minll+DPO" stick together, same for "Unll" and "Unll+DPO", "Contrastll" and "Contrastll+DPO"
bar_positions = [-0.2, 0.4, 1, 2, 3.2, 1.3, 2.3, 3.5]  # 2/2.3, 3/3.3, 4/4.3 are close

fig, axs = plt.subplots(figsize=(7.5, 5))
bars = axs.bar(bar_positions, result, width=0.3, color='skyblue')

# Set x-ticks at the center of each group
axs.set_xticks([-0.2, 0.4, 1.15, 2.15, 3.35], ["SFT", "DPO", "Minll/Minll+DPO", "Unll/Unll+DPO", "Contrastll/Contrastll+DPO"])

# Annotate each bar with its value
for bar, value in zip(bars, result):
    axs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.3f}", ha='center', va='bottom')
    # Change color of specific bars to red
highlight_indices = [5, 6, 7]  # Indices for "Minll+DPO", "Unll+DPO", "Contrastll+DPO"
for idx in highlight_indices:
    bars[idx].set_color("#217449")  # Change color to green
for idx in [0]:
    bars[idx].set_color("#383ab3f4")
for idx in [1]:
    bars[idx].set_color("#be2525f4")
for idx in [2,3,4]:
    bars[idx].set_color("#15E67A")  # Change color to skyblue for others
axs.set_ylabel("Eval win rate")

axs.set_ylim(0.4, 1.05)  # Set y-axis limit to make it more visually appealing
fig.savefig("../output/leaderboard_plot.png", bbox_inches='tight', dpi=800)