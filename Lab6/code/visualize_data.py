import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
train_df = pd.read_csv('datasets/traffic_forecasting/processed_train.csv', parse_dates=['Datetime'])
test_df = pd.read_csv('datasets/traffic_forecasting/processed_test.csv', parse_dates=['Datetime'])

train_df.set_index('Datetime', inplace=True)
test_df.set_index('Datetime', inplace=True)

print(f"Train data: {len(train_df)} samples from {train_df.index[0]} to {train_df.index[-1]}")
print(f"Test data:  {len(test_df)} samples from {test_df.index[0]} to {test_df.index[-1]}")

# ============================================================================
# VISUALIZATION 1: FULL TIME SERIES
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Plot train data
ax.plot(train_df.index, train_df['Total'], 
        color='#3498db', linewidth=1.5, label='Train Data', alpha=0.8)

# Plot test data
ax.plot(test_df.index, test_df['Total'], 
        color='#e74c3c', linewidth=1.5, label='Test Data', alpha=0.8)

# Add vertical line at split point
split_point = test_df.index[0]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=2, 
           label=f'Train/Test Split', alpha=0.7)

# Formatting
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Total (Traffic Value)', fontsize=12, fontweight='bold')
ax.set_title('Traffic Time Series: Train vs Test Split', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_full_series.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: train_test_full_series.png")

# ============================================================================
# VISUALIZATION 2: ZOOMED VIEW OF SPLIT BOUNDARY
# ============================================================================

# Get last 500 points of train and first 500 points of test
n_points = min(500, len(train_df), len(test_df))
train_tail = train_df.tail(n_points)
test_head = test_df.head(n_points)

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(train_tail.index, train_tail['Total'], 
        color='#3498db', linewidth=2, label='Train Data (End)', alpha=0.8, marker='o', markersize=3)

ax.plot(test_head.index, test_head['Total'], 
        color='#e74c3c', linewidth=2, label='Test Data (Start)', alpha=0.8, marker='s', markersize=3)

# Add vertical line at split point
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=2, 
           label=f'Train/Test Split', alpha=0.7)

# Formatting
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Total (Traffic Value)', fontsize=12, fontweight='bold')
ax.set_title('Traffic Time Series: Zoomed View Around Train/Test Split', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_split_zoomed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_split_zoomed.png")

# ============================================================================
# VISUALIZATION 3: STATISTICAL COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograms
axes[0, 0].hist(train_df['Total'], bins=50, alpha=0.6, color='#3498db', label='Train', density=True)
axes[0, 0].hist(test_df['Total'], bins=50, alpha=0.6, color='#e74c3c', label='Test', density=True)
axes[0, 0].set_xlabel('Total (Traffic Value)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plots
data_to_plot = [train_df['Total'].values, test_df['Total'].values]
bp = axes[0, 1].boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[0, 1].set_ylabel('Total (Traffic Value)')
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Rolling mean (shows trends)
window = 96  # One day if periodicity is 96
train_rolling = train_df['Total'].rolling(window=window).mean()
test_rolling = test_df['Total'].rolling(window=window).mean()

axes[1, 0].plot(train_df.index, train_rolling, color='#3498db', linewidth=2, label='Train (Rolling Mean)', alpha=0.8)
axes[1, 0].plot(test_df.index, test_rolling, color='#e74c3c', linewidth=2, label='Test (Rolling Mean)', alpha=0.8)
axes[1, 0].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
axes[1, 0].set_xlabel('Datetime')
axes[1, 0].set_ylabel('Total (Traffic Value)')
axes[1, 0].set_title(f'Rolling Mean (window={window})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Statistics table
stats_train = train_df['Total'].describe()
stats_test = test_df['Total'].describe()

stats_text = f"""
TRAIN DATA:
  Count:  {stats_train['count']:.0f}
  Mean:   {stats_train['mean']:.4f}
  Std:    {stats_train['std']:.4f}
  Min:    {stats_train['min']:.4f}
  25%:    {stats_train['25%']:.4f}
  Median: {stats_train['50%']:.4f}
  75%:    {stats_train['75%']:.4f}
  Max:    {stats_train['max']:.4f}

TEST DATA:
  Count:  {stats_test['count']:.0f}
  Mean:   {stats_test['mean']:.4f}
  Std:    {stats_test['std']:.4f}
  Min:    {stats_test['min']:.4f}
  25%:    {stats_test['25%']:.4f}
  Median: {stats_test['50%']:.4f}
  75%:    {stats_test['75%']:.4f}
  Max:    {stats_test['max']:.4f}
"""

axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')
axes[1, 1].set_title('Statistical Summary')

plt.tight_layout()
plt.savefig('train_test_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_statistics.png")

# ============================================================================
# VISUALIZATION 4: DAILY PATTERN (if periodicity = 96)
# ============================================================================

# Reshape data to see daily patterns (assuming 96 = one day)
periodicity = 96

# Take a few days from train and test
n_days_to_show = 7
train_sample = train_df.tail(periodicity * n_days_to_show)
test_sample = test_df.head(periodicity * n_days_to_show)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot train sample
for day in range(n_days_to_show):
    start_idx = day * periodicity
    end_idx = start_idx + periodicity
    day_data = train_sample.iloc[start_idx:end_idx]
    axes[0].plot(range(periodicity), day_data['Total'].values, 
                alpha=0.6, linewidth=1.5, label=f'Day {day+1}')

axes[0].set_xlabel('Time Step (within day)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Total (Traffic Value)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Train Data: Daily Patterns (Last {n_days_to_show} days)', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', ncol=7)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, periodicity-1)

# Plot test sample
for day in range(min(n_days_to_show, len(test_sample) // periodicity)):
    start_idx = day * periodicity
    end_idx = start_idx + periodicity
    if end_idx <= len(test_sample):
        day_data = test_sample.iloc[start_idx:end_idx]
        axes[1].plot(range(periodicity), day_data['Total'].values, 
                    alpha=0.6, linewidth=1.5, label=f'Day {day+1}')

axes[1].set_xlabel('Time Step (within day)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total (Traffic Value)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Test Data: Daily Patterns (First {n_days_to_show} days)', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', ncol=7)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, periodicity-1)

plt.tight_layout()
plt.savefig('train_test_daily_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_daily_patterns.png")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTrain Data ({len(train_df)} samples):")
print(f"  Range: {train_df.index[0]} to {train_df.index[-1]}")
print(f"  Mean:  {stats_train['mean']:.4f}")
print(f"  Std:   {stats_train['std']:.4f}")
print(f"  Min:   {stats_train['min']:.4f}")
print(f"  Max:   {stats_train['max']:.4f}")

print(f"\nTest Data ({len(test_df)} samples):")
print(f"  Range: {test_df.index[0]} to {test_df.index[-1]}")
print(f"  Mean:  {stats_test['mean']:.4f}")
print(f"  Std:   {stats_test['std']:.4f}")
print(f"  Min:   {stats_test['min']:.4f}")
print(f"  Max:   {stats_test['max']:.4f}")

# Check for data shift
mean_diff = abs(stats_train['mean'] - stats_test['mean'])
std_diff = abs(stats_train['std'] - stats_test['std'])

print(f"\nDistribution Comparison:")
print(f"  Mean difference: {mean_diff:.4f} ({mean_diff/stats_train['mean']*100:.2f}%)")
print(f"  Std difference:  {std_diff:.4f} ({std_diff/stats_train['std']*100:.2f}%)")

if mean_diff/stats_train['mean'] > 0.1:
    print("\n  ⚠️  WARNING: Significant mean shift detected between train and test!")
    print("     This could make prediction harder.")

if std_diff/stats_train['std'] > 0.2:
    print("\n  ⚠️  WARNING: Significant variance shift detected between train and test!")
    print("     The test data has different variability than training data.")

print("\n" + "="*70)
print("VISUALIZATION FILES CREATED:")
print("="*70)
print("  1. train_test_full_series.png     - Full time series overview")
print("  2. train_test_split_zoomed.png    - Zoomed view around split")
print("  3. train_test_statistics.png      - Statistical comparison")
print("  4. train_test_daily_patterns.png  - Daily pattern analysis")
print("="*70)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
train_df = pd.read_csv('datasets/traffic_forecasting/processed_train.csv', parse_dates=['Datetime'])
test_df = pd.read_csv('datasets/traffic_forecasting/processed_test.csv', parse_dates=['Datetime'])

train_df.set_index('Datetime', inplace=True)
test_df.set_index('Datetime', inplace=True)

print(f"Train data: {len(train_df)} samples from {train_df.index[0]} to {train_df.index[-1]}")
print(f"Test data:  {len(test_df)} samples from {test_df.index[0]} to {test_df.index[-1]}")

# ============================================================================
# VISUALIZATION 1: FULL TIME SERIES
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Plot train data
ax.plot(train_df.index, train_df['Total'], 
        color='#3498db', linewidth=1.5, label='Train Data', alpha=0.8)

# Plot test data
ax.plot(test_df.index, test_df['Total'], 
        color='#e74c3c', linewidth=1.5, label='Test Data', alpha=0.8)

# Add vertical line at split point
split_point = test_df.index[0]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=2, 
           label=f'Train/Test Split', alpha=0.7)

# Formatting
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Total (Traffic Value)', fontsize=12, fontweight='bold')
ax.set_title('Traffic Time Series: Train vs Test Split', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_full_series.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: train_test_full_series.png")

# ============================================================================
# VISUALIZATION 2: ZOOMED VIEW OF SPLIT BOUNDARY
# ============================================================================

# Get last 500 points of train and first 500 points of test
n_points = min(500, len(train_df), len(test_df))
train_tail = train_df.tail(n_points)
test_head = test_df.head(n_points)

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(train_tail.index, train_tail['Total'], 
        color='#3498db', linewidth=2, label='Train Data (End)', alpha=0.8, marker='o', markersize=3)

ax.plot(test_head.index, test_head['Total'], 
        color='#e74c3c', linewidth=2, label='Test Data (Start)', alpha=0.8, marker='s', markersize=3)

# Add vertical line at split point
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=2, 
           label=f'Train/Test Split', alpha=0.7)

# Formatting
ax.set_xlabel('Datetime', fontsize=12, fontweight='bold')
ax.set_ylabel('Total (Traffic Value)', fontsize=12, fontweight='bold')
ax.set_title('Traffic Time Series: Zoomed View Around Train/Test Split', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_split_zoomed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_split_zoomed.png")

# ============================================================================
# VISUALIZATION 3: STATISTICAL COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograms
axes[0, 0].hist(train_df['Total'], bins=50, alpha=0.6, color='#3498db', label='Train', density=True)
axes[0, 0].hist(test_df['Total'], bins=50, alpha=0.6, color='#e74c3c', label='Test', density=True)
axes[0, 0].set_xlabel('Total (Traffic Value)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plots
data_to_plot = [train_df['Total'].values, test_df['Total'].values]
bp = axes[0, 1].boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
axes[0, 1].set_ylabel('Total (Traffic Value)')
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Rolling mean (shows trends)
window = 96  # One day if periodicity is 96
train_rolling = train_df['Total'].rolling(window=window).mean()
test_rolling = test_df['Total'].rolling(window=window).mean()

axes[1, 0].plot(train_df.index, train_rolling, color='#3498db', linewidth=2, label='Train (Rolling Mean)', alpha=0.8)
axes[1, 0].plot(test_df.index, test_rolling, color='#e74c3c', linewidth=2, label='Test (Rolling Mean)', alpha=0.8)
axes[1, 0].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
axes[1, 0].set_xlabel('Datetime')
axes[1, 0].set_ylabel('Total (Traffic Value)')
axes[1, 0].set_title(f'Rolling Mean (window={window})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Statistics table
stats_train = train_df['Total'].describe()
stats_test = test_df['Total'].describe()

stats_text = f"""
TRAIN DATA:
  Count:  {stats_train['count']:.0f}
  Mean:   {stats_train['mean']:.4f}
  Std:    {stats_train['std']:.4f}
  Min:    {stats_train['min']:.4f}
  25%:    {stats_train['25%']:.4f}
  Median: {stats_train['50%']:.4f}
  75%:    {stats_train['75%']:.4f}
  Max:    {stats_train['max']:.4f}

TEST DATA:
  Count:  {stats_test['count']:.0f}
  Mean:   {stats_test['mean']:.4f}
  Std:    {stats_test['std']:.4f}
  Min:    {stats_test['min']:.4f}
  25%:    {stats_test['25%']:.4f}
  Median: {stats_test['50%']:.4f}
  75%:    {stats_test['75%']:.4f}
  Max:    {stats_test['max']:.4f}
"""

axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')
axes[1, 1].set_title('Statistical Summary')

plt.tight_layout()
plt.savefig('train_test_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_statistics.png")

# ============================================================================
# VISUALIZATION 4: DAILY PATTERN (if periodicity = 96)
# ============================================================================

# Reshape data to see daily patterns (assuming 96 = one day)
periodicity = 96

# Take a few days from train and test
n_days_to_show = 7
train_sample = train_df.tail(periodicity * n_days_to_show)
test_sample = test_df.head(periodicity * n_days_to_show)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot train sample
for day in range(n_days_to_show):
    start_idx = day * periodicity
    end_idx = start_idx + periodicity
    day_data = train_sample.iloc[start_idx:end_idx]
    axes[0].plot(range(periodicity), day_data['Total'].values, 
                alpha=0.6, linewidth=1.5, label=f'Day {day+1}')

axes[0].set_xlabel('Time Step (within day)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Total (Traffic Value)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Train Data: Daily Patterns (Last {n_days_to_show} days)', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', ncol=7)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, periodicity-1)

# Plot test sample
for day in range(min(n_days_to_show, len(test_sample) // periodicity)):
    start_idx = day * periodicity
    end_idx = start_idx + periodicity
    if end_idx <= len(test_sample):
        day_data = test_sample.iloc[start_idx:end_idx]
        axes[1].plot(range(periodicity), day_data['Total'].values, 
                    alpha=0.6, linewidth=1.5, label=f'Day {day+1}')

axes[1].set_xlabel('Time Step (within day)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total (Traffic Value)', fontsize=11, fontweight='bold')
axes[1].set_title(f'Test Data: Daily Patterns (First {n_days_to_show} days)', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', ncol=7)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, periodicity-1)

plt.tight_layout()
plt.savefig('train_test_daily_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: train_test_daily_patterns.png")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTrain Data ({len(train_df)} samples):")
print(f"  Range: {train_df.index[0]} to {train_df.index[-1]}")
print(f"  Mean:  {stats_train['mean']:.4f}")
print(f"  Std:   {stats_train['std']:.4f}")
print(f"  Min:   {stats_train['min']:.4f}")
print(f"  Max:   {stats_train['max']:.4f}")

print(f"\nTest Data ({len(test_df)} samples):")
print(f"  Range: {test_df.index[0]} to {test_df.index[-1]}")
print(f"  Mean:  {stats_test['mean']:.4f}")
print(f"  Std:   {stats_test['std']:.4f}")
print(f"  Min:   {stats_test['min']:.4f}")
print(f"  Max:   {stats_test['max']:.4f}")

# Check for data shift
mean_diff = abs(stats_train['mean'] - stats_test['mean'])
std_diff = abs(stats_train['std'] - stats_test['std'])

print(f"\nDistribution Comparison:")
print(f"  Mean difference: {mean_diff:.4f} ({mean_diff/stats_train['mean']*100:.2f}%)")
print(f"  Std difference:  {std_diff:.4f} ({std_diff/stats_train['std']*100:.2f}%)")

if mean_diff/stats_train['mean'] > 0.1:
    print("\n  ⚠️  WARNING: Significant mean shift detected between train and test!")
    print("     This could make prediction harder.")

if std_diff/stats_train['std'] > 0.2:
    print("\n  ⚠️  WARNING: Significant variance shift detected between train and test!")
    print("     The test data has different variability than training data.")

print("\n" + "="*70)
print("VISUALIZATION FILES CREATED:")
print("="*70)
print("  1. train_test_full_series.png     - Full time series overview")
print("  2. train_test_split_zoomed.png    - Zoomed view around split")
print("  3. train_test_statistics.png      - Statistical comparison")
print("  4. train_test_daily_patterns.png  - Daily pattern analysis")
print("="*70)

plt.show()