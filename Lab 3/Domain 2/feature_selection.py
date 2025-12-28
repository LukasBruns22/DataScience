import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pipeline import run_step_comparison

# --- APPROACH 1: Statistical (Chi-Squared) ---
def app1_chi2(X_train, y_train, X_test, y_test):
    print("      (Selecting Top 15 Features using Chi2...)")
    
    # 1. Check for negatives (Chi2 fails with negatives)
    # Since MinMax Scaling won previously, we should be safe. 
    # But if there's a tiny negative float error, we clip it.
    X_train = X_train.clip(lower=0) 
    
    # Select top 15 features (You can tweak 'k')
    selector = SelectKBest(score_func=chi2, k=9)
    selector.fit(X_train, y_train)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> Selected: {list(cols)}")
    
    # Transform
    X_train_sel = X_train[cols]
    X_test_sel = X_test[cols]
    
    return X_train_sel, y_train, X_test_sel, y_test

# --- APPROACH 2: Model-Based (Random Forest Importance) ---
def app2_random_forest(X_train, y_train, X_test, y_test):
    print("      (Selecting features using Random Forest importance...)")
    
    # Train a quick tree to find importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Select features that are more important than the average feature
    selector = SelectFromModel(rf, prefit=True)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> RF kept {len(cols)} features: {list(cols)}")
    
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