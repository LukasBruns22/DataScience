from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pipeline import run_step_comparison

# --- APPROACH 1: SMOTE (Oversampling) ---
def app1_smote(X_train, y_train, X_test, y_test):
    print("      (Applying SMOTE... this may take a moment)")
    smote = SMOTE(random_state=42)
    
    # Resample TRAIN only
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, y_train_res, X_test, y_test

# --- APPROACH 2: Random Undersampling ---
def app2_undersample(X_train, y_train, X_test, y_test):
    rus = RandomUnderSampler(random_state=42)
    
    # Resample TRAIN only
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    
    return X_train_res, y_train_res, X_test, y_test

def run_balancing_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_smote,
        app2_func=app2_undersample,
        step_name="BALANCING"
    )