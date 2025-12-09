import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data

df = load_data()
target_col = "Inflation Rate (%)"

print(f"Analyzing granularity for variable: {target_col}")


print("--- GRANULARITY TRANSFORMATION ---")

# 1. Original Frequency
print(f"Original Frequency detected: M")

# 2. Quarterly (Trimestriel)
df_quarterly = df.resample('QE').mean()
print(f"Quarterly data shape: {df_quarterly.shape}")

# 3. Yearly (Annuel)
df_yearly = df.resample('YE').mean()
print(f"Yearly data shape: {df_yearly.shape}")


sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

fig.suptitle(f'Granularity Analysis: {target_col} (USA)', fontsize=16)

# Plot 1: Original
axes[0].plot(df.index, df[target_col], color='navy', alpha=0.6, linewidth=1)
axes[0].set_title(f'1. Original Frequency (M)', fontsize=12)
axes[0].set_ylabel('Inflation Rate %')
axes[0].set_xlabel('Month')

# Plot 2: Quarterly
axes[1].plot(df_quarterly.index, df_quarterly[target_col], color='darkorange', marker='o', markersize=4, linewidth=2)
axes[1].set_title('2. Quarterly Resampling', fontsize=12)
axes[1].set_ylabel('Inflation Rate %')
axes[1].set_xlabel('Quarter')

# Plot 3: Yearly
axes[2].plot(df_yearly.index, df_yearly[target_col], color='green', marker='s', markersize=6, linewidth=2)
axes[2].set_title('3. Yearly Resampling', fontsize=12)
axes[2].set_ylabel('Inflation Rate %')
axes[2].set_xlabel('Year')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Lab5/eco/results/2_granularity.png')
plt.close()