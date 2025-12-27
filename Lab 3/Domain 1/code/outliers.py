import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats
from pipeline import run_step_comparison

# --- APPROACH 1: Isolation Forest ---
def app1_isolation_forest(X_train, y_train, X_test, y_test):
    print("      (Running Isolation Forest...)")
    
    iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X_train)
    
    mask = preds == 1
    print(f"      -> Removed {sum(~mask)} outliers ({sum(~mask)/len(mask)*100:.1f}% of data).")
    return X_train[mask], y_train[mask], X_test, y_test

# --- APPROACH 2: Z-Score ---
def app2_zscore(X_train, y_train, X_test, y_test):
    print("      (Running Robust Z-Score...)")
    
    std_devs = X_train.std()
    valid_cols = std_devs[std_devs > 1e-9].index 
    
    if len(valid_cols) == 0:
        print("      [!] Warning: No variable columns found. Skipping Z-Score.")
        return X_train, y_train, X_test, y_test

    # Calculate Z-Scores ONLY on valid columns
    subset = X_train[valid_cols]
    z_scores = np.abs(stats.zscore(subset))
    
    extreme_per_row = (z_scores > 5).sum(axis=1)
    mask = extreme_per_row == 0 
    
    rows_removed = sum(~mask)
    pct_removed = rows_removed / len(mask) * 100
    
    # Safety check: Don't remove more than 5% of data
    if pct_removed > 5:
        print(f"      [!] Z-Score would remove {pct_removed:.1f}% of data - too aggressive!")
        print(f"      -> Keeping only the most extreme 2% as outliers")
        # Instead, just remove the top 2% most extreme rows
        max_z_per_row = z_scores.max(axis=1)
        threshold = np.percentile(max_z_per_row, 98)
        mask = max_z_per_row < threshold
        rows_removed = sum(~mask)
    
    print(f"      -> Removed {rows_removed} outliers ({rows_removed/len(mask)*100:.1f}% of data).")
    return X_train[mask], y_train[mask], X_test, y_test

def run_outliers_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_isolation_forest,
        app2_func=app2_zscore,
        step_name="OUTLIERS TREATMENT"
    )