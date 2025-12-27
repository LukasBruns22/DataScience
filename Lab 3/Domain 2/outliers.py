import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
from pipeline import run_step_comparison

# --- APPROACH 1: Isolation Forest (No changes needed) ---
def app1_isolation_forest(X_train, y_train, X_test, y_test):
    print("      (Running Isolation Forest...)")
    # Contamination 1%
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X_train)
    
    # Keep only inliers (1)
    mask = preds == 1
    print(f"      -> Removed {sum(~mask)} outliers.")
    return X_train[mask], y_train[mask], X_test, y_test

# --- APPROACH 2: Robust Z-Score (The Fix) ---
def app2_zscore(X_train, y_train, X_test, y_test):
    print("      (Running Robust Z-Score...)")
    
    # 1. Identify columns that actually have variance (std > 0)
    # If a column is constant (0 variance), Z-score explodes to Infinity. We must skip those.
    std_devs = X_train.std()
    # Keep columns where std is not effectively zero
    valid_cols = std_devs[std_devs > 1e-9].index 
    
    if len(valid_cols) == 0:
        print("      [!] Warning: No variable columns found. Skipping Z-Score.")
        return X_train, y_train, X_test, y_test

    # 2. Calculate Z-Scores ONLY on valid columns
    subset = X_train[valid_cols]
    z_scores = np.abs(stats.zscore(subset))
    
    # 3. Filter: Keep rows where ALL valid columns are < 3 SD
    # (nan_policy='omit' isn't needed if we pre-filter constant columns)
    mask = (z_scores < 3).all(axis=1)
    
    # 4. FAIL-SAFE: If we removed everything, return original
    rows_remaining = mask.sum()
    if rows_remaining == 0:
        print("      [!] Error: Z-Score was too aggressive and removed ALL rows.")
        print("      -> Fallback: Keeping original dataset.")
        return X_train, y_train, X_test, y_test
    
    if rows_remaining < len(X_train) * 0.5:
         print(f"      [!] Warning: Z-Score removed >50% of data ({sum(~mask)} rows).")
         # Optional: You could return original here too if you feel 50% loss is too much
    
    print(f"      -> Removed {sum(~mask)} outliers.")
    return X_train[mask], y_train[mask], X_test, y_test

def run_outliers_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_isolation_forest,
        app2_func=app2_zscore,
        step_name="OUTLIERS TREATMENT"
    )