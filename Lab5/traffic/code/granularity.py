import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data

# Visual setup
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (15, 12)

df = load_data()
target_col = 'Total'


# B. Hourly (1H) 
df_hourly = df[target_col].resample('h').sum()

# C. Daily (1D) 
df_daily = df[target_col].resample('D').sum()


fig, axes = plt.subplots(3, 1, sharex=False)
fig.suptitle(f'Granularity Analysis: Traffic Volume ({target_col})', fontsize=16)

# Plot 1: Original (High Frequency)
axes[0].plot(df.index, df[target_col], color='#1f77b4', linewidth=0.5, alpha=0.8)
axes[0].set_title('1. Original Granularity (15 min)', fontsize=12)
axes[0].set_ylabel('Traffic / 15 min')

# Plot 2: Hourly (Medium Frequency)
axes[1].plot(df_hourly.index, df_hourly, color='#ff7f0e', linewidth=1)
axes[1].set_title('2. Hourly Granularity (1H)', fontsize=12)
axes[1].set_ylabel('Traffic / Hour')

# Plot 3: Daily (Low Frequency)
axes[2].plot(df_daily.index, df_daily, color='#2ca02c', marker='o', linewidth=2)
axes[2].set_title('3. Daily Granularity (1D)', fontsize=12)
axes[2].set_ylabel('Traffic / Day')
axes[2].set_xlabel('Date')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Lab5/traffic/results/2_granularity.png')
plt.close() 