# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/eval_metrics.csv")

# Boxplot of returns
plt.figure()
df.boxplot(column="return", by="policy")
plt.title("Episode Profit by Policy")
plt.suptitle("")
plt.ylabel("Profit")
plt.savefig("runs/box_return.png", dpi=150)

# Boxplot of fill rate (service level)
plt.figure()
df.boxplot(column="fill_rate", by="policy")
plt.title("Service Level (Fill Rate) by Policy")
plt.suptitle("")
plt.ylabel("Fill Rate")
plt.ylim(0, 1.05)
plt.savefig("runs/box_fillrate.png", dpi=150)

print("Saved: runs/box_return.png, runs/box_fillrate.png")
