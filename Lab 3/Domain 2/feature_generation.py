import pandas as pd
import numpy as np
from pipeline import run_step_comparison

# --- APPROACH 1: Physics (Speed & Log) ---
# (This remains unchanged, it works fine)
def app1_physics(X_train, y_train, X_test, y_test):
    print("      (Generating Physics: Speed & Log Transforms...)")
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # 1. Speed Proxy (Dist / Time)
    # Use 'CRSElapsedTime' (Flight Duration) which should still be in the dataset
    # If CRSElapsedTime was also encoded/dropped, check your columns. 
    # Usually Duration is linear (not cyclical) so it should exist.
    if 'CRSElapsedTime' in X_train.columns:
        X_train_new['Speed_Proxy'] = X_train['Distance'] / (X_train['CRSElapsedTime'] + 1)
        X_test_new['Speed_Proxy'] = X_test['Distance'] / (X_test['CRSElapsedTime'] + 1)
    
    # 2. Log Distance
    X_train_new['Log_Distance'] = np.log1p(X_train['Distance'])
    X_test_new['Log_Distance'] = np.log1p(X_test['Distance'])
    
    return X_train_new, y_train, X_test_new, y_test

# --- APPROACH 2: Context (Quadrants & Risk) ---
def app2_context_bins(X_train, y_train, X_test, y_test):
    print("      (Generating Context: Time Quadrants, Seasons, Risk...)")
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # 1. Time of Day (Using Sin/Cos Quadrants)
    # We create a function to map sin/cos back to 4 buckets
    def get_quadrant(row, prefix):
        sin_val = row.get(f'{prefix}_sin', 0)
        cos_val = row.get(f'{prefix}_cos', 0)
        
        # Logic matches the unit circle mapping of 24h clock
        if sin_val >= 0 and cos_val >= 0: return 0  # Night (0-6)
        elif sin_val >= 0 and cos_val < 0: return 1 # Morning (6-12)
        elif sin_val < 0 and cos_val < 0:  return 2 # Afternoon (12-18)
        else: return 3                              # Evening (18-24)

    # Apply to Train and Test
    if 'CRSDepTime_sin' in X_train.columns:
        X_train_new['Time_Bin'] = X_train.apply(lambda row: get_quadrant(row, 'CRSDepTime'), axis=1)
        X_test_new['Time_Bin'] = X_test.apply(lambda row: get_quadrant(row, 'CRSDepTime'), axis=1)

    # 2. Seasonality (Using Month_cos)
    # Cosine of Month is a great proxy for Temperature/Season
    # Jan/Dec (Winter) -> Cos approx 1.0
    # July (Summer)    -> Cos approx -1.0
    if 'Month_cos' in X_train.columns:
        # We simply create a "Is_Winter" or "Temperature_Proxy" feature
        # If Cos > 0.5, it's deep winter. If Cos < -0.5, it's deep summer.
        # Let's just keep the raw Month_cos as the "Season" feature since it is continuous.
        # But if you want bins:
        X_train_new['Is_Winter'] = (X_train['Month_cos'] > 0.5).astype(int)
        X_test_new['Is_Winter'] = (X_test['Month_cos'] > 0.5).astype(int)
    
    # 3. Risk Interactions
    if 'Hub_Airline' in X_train.columns:
        X_train_new['Hub_x_Dest'] = X_train['Hub_Airline'] * X_train['Dest']
        X_test_new['Hub_x_Dest'] = X_test['Hub_Airline'] * X_test['Dest']

    return X_train_new, y_train, X_test_new, y_test

def run_feature_generation_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_physics,
        app2_func=app2_context_bins,
        step_name="FEATURE GENERATION"
    )