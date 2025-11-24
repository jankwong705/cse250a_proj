import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df = pd.read_csv('hmm_tuning_results.csv')

# Set style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='threshold_minutes', y='accuracy', hue='n_components', marker='o', palette='viridis')

# Customize
plt.title('HMM Accuracy vs. Session Threshold and Number of Components', fontsize=16)
plt.xlabel('Session Threshold (minutes)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks([10, 20, 30, 60, 120])
plt.legend(title='n_components')

# Save
plt.tight_layout()
plt.savefig('tuning_results.png', dpi=300)
print("Plot saved to tuning_results.png")
