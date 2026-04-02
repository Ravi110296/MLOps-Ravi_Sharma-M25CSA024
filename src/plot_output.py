import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("fgsm_results.csv")

# Plot
plt.figure()
plt.plot(df["epsilon"], df["accuracy"], marker='o')

plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("FGSM: Epsilon vs Accuracy")
plt.grid(True)

# Save
plt.savefig("fgsm_plot.png")
plt.close()

print("✅ FGSM plot saved as fgsm_plot.png")