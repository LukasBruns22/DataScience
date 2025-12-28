from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pipeline import run_step_comparison
import pandas as pd

# --- APPROACH 1: MinMax Scaler (0 to 1) ---
def app1_minmax(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    # Fit on TRAIN, Transform BOTH (Standard Leakage Prevention)
    cols = X_train.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

# --- APPROACH 2: Standard Scaler (Z-Score Normalization) ---
def app2_standard(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    
    cols = X_train.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def run_scaling_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_minmax,
        app2_func=app2_standard,
        step_name="SCALING"
    )