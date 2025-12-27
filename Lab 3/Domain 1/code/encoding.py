import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Load dataset
df = pd.read_csv('datasets/traffic_accidents/traffic_accidents_no_leakage.csv')

# 1. HANDLE TARGET
target_map = {
    'NO INJURY / DRIVE AWAY': 0,
    'INJURY AND / OR TOW DUE TO CRASH': 1
}
df['crash_type_enc'] = df['crash_type'].map(target_map)

# 2. HANDLE UNKNOWNS
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna('UNKNOWN')

# 3. ENCODING

# A. Binary
df['intersection_related_i'] = df['intersection_related_i'].map({'Y': 1, 'N': 0})

# B. TARGET ENCODING (WITH PROPER CROSS-VALIDATION TO AVOID LEAKAGE!)
def target_encode_with_cv(df, col, target, n_splits=5, smoothing=10):
    """
    Target encode with cross-validation to prevent leakage
    smoothing: controls how much we trust small sample sizes
    """
    encoded = pd.Series(index=df.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    global_mean = df[target].mean()
    
    for train_idx, val_idx in skf.split(df, df[target]):
        # Calculate means only on training fold
        train_df = df.iloc[train_idx]
        
        # Calculate statistics for each category
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        
        # Apply smoothing: blend category mean with global mean
        # More samples = trust category mean more
        stats['smoothed'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) / 
            (stats['count'] + smoothing)
        )
        
        # Map to validation fold
        encoded.iloc[val_idx] = df.iloc[val_idx][col].map(stats['smoothed'])
    
    # Fill any remaining NaNs with global mean
    encoded = encoded.fillna(global_mean)
    
    return encoded

# Apply target encoding to high-cardinality categorical features
print("Applying target encoding to prim_contributory_cause...")
df['prim_contributory_cause_enc'] = target_encode_with_cv(
    df, 'prim_contributory_cause', 'crash_type_enc'
)

# C. Cyclic Encoding
def encode_cyclic(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    return data

print("Applying cyclic encoding...")
df = encode_cyclic(df, 'crash_hour', 24)
df = encode_cyclic(df, 'crash_day_of_week', 7)
df = encode_cyclic(df, 'crash_month', 12)

# D. ONE-HOT ENCODING (More selective grouping)
one_hot_cols = [
    'traffic_control_device', 'weather_condition', 'lighting_condition',
    'first_crash_type', 'trafficway_type', 'alignment', 
    'roadway_surface_cond', 'road_defect'
]

# Less aggressive rare label grouping (0.5% threshold instead of 1%)
def group_rare_labels(df, col, threshold=0.005):
    counts = df[col].value_counts(normalize=True)
    valid_labels = counts[counts >= threshold].index
    df[col] = df[col].apply(lambda x: x if x in valid_labels else 'OTHER')
    return df

print("Applying one-hot encoding...")
for col in one_hot_cols:
    df = group_rare_labels(df, col)

# Apply One-Hot
df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# 4. Extract temporal info before dropping crash_date
if 'crash_date' in df.columns:
    df_encoded['crash_date_dt'] = pd.to_datetime(df_encoded['crash_date'])
    min_date = df_encoded['crash_date_dt'].min()
    df_encoded['days_since_start'] = (df_encoded['crash_date_dt'] - min_date).dt.days
    df_encoded = df_encoded.drop(['crash_date', 'crash_date_dt'], axis=1)

# 5. CLEANUP
cols_to_drop = [
    'prim_contributory_cause', 'crash_type', 'crash_hour', 
    'crash_day_of_week', 'crash_month'
]
df_encoded = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])

print("\n=== Encoding Summary ===")
print(f"Final shape: {df_encoded.shape}")
print(f"Columns: target_encoded, cyclic (sin/cos), one-hot dummies, days_since_start")

df_encoded.to_csv('datasets/traffic_accidents/traffic_accidents_encoded.csv', index=False)
print("\nSaved to: traffic_accidents_encoded.csv")