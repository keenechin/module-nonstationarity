import matplotlib.pyplot as plt
import numpy as np

labels = ["TPU -> TPU", "Silicone -> Silicone", "TPU -> Silicone", "Silicone -> TPU"]
time = [20.09, 43.44, 32.27, 39.44]
steps = [4.4, 10.5, 7.5, 9.13]
reward = [11.64, 2.24, -0.82, 0.09]
success = [1, 0.4, 1, 0.8]

x = np.arange(len(labels))
width = 0.35
fig, axes = plt.subplots(nrows=2)

rects1 = axes[0].bar(x-width/2, time, width, label="Mean time to complete")
rects2 = axes[0].bar(x+width/2, steps, width, label="Mean steps to complete")

rects3 = axes[1].bar(x-width/2, reward, width, label="Mean Reward")
rects4 = axes[1].bar(x+width/2, success, width, label="Success Ratio")

for ax in axes:
    ax.set_xticks(x, labels)
    ax.legend()
    
axes[0].bar_label(rects1, padding=0)
axes[0].bar_label(rects2, padding=0)
axes[1].bar_label(rects3, padding=0)
axes[1].bar_label(rects4, padding=0)

fig.tight_layout()
plt.show()