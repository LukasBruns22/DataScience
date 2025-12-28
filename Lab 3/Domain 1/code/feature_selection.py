import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pipeline import run_step_comparison

# --- APPROACH 1: Keep More Features (Chi-Squared) ---
def app1_chi2(X_train, y_train, X_test, y_test):
    print("      (Selecting Top Features using Chi2...)")
    
    # Clip negatives for Chi2
    X_train_clip = X_train.clip(lower=0)
    X_test_clip = X_test.clip(lower=0)
    
    # Select top 50% of features
    k = max(15, int(X_train.shape[1] * 0.5))
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train_clip, y_train)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> Selected {len(cols)} features (from {X_train.shape[1]})")
    
    # Transform
    X_train_sel = X_train[cols]
    X_test_sel = X_test[cols]
    
    return X_train_sel, y_train, X_test_sel, y_test

# --- APPROACH 2: Random Forest ---
def app2_random_forest(X_train, y_train, X_test, y_test):
    print("      (Selecting features using Random Forest importance...)")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Keep features with importance > median
    threshold = np.median(importances)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> RF kept {len(cols)} features (from {X_train.shape[1]})")
    
    # If we kept too few, just keep top 50%
    if len(cols) < X_train.shape[1] * 0.3:
        print(f"      -> Too few features, keeping top 50% by importance instead")
        indices = np.argsort(importances)[::-1][:int(X_train.shape[1] * 0.5)]
        cols = X_train.columns[indices]
    
    X_train_sel = X_train[cols]
    X_test_sel = X_test[cols]
    
    return X_train_sel, y_train, X_test_sel, y_test

def run_feature_selection_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_chi2,
        app2_func=app2_random_forest,
        step_name="FEATURE SELECTION"
    )