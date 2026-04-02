import matplotlib.pyplot as plt

attacks = ["PGD", "BIM"]
acc = [98.15, 97.425]

plt.figure(figsize=(6, 4))

bars = plt.bar(attacks, acc, color=["steelblue", "orange"])

# Labels
plt.ylabel("Detection Accuracy (%)", fontsize=12)
plt.title("Detection Performance Comparison", fontsize=14)

# Y-axis limits for better visualization
plt.ylim(90, 100)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.2,
             f"{height:.2f}%",
             ha='center', fontsize=11)

# Grid for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tight layout
plt.tight_layout()

# Save high-quality image
plt.savefig("detection_comparison.png", dpi=300)
plt.close()

print("✅ Styled plot saved!")