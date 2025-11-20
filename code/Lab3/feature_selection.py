import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def approach_model_based_selection(X_train, X_test, y_train, n_features=10):
    """
    Approach 1: Feature Selection using a Random Forest Classifier.
    Selects the N most important features.
    """
    X_train_sel = X_train.copy()
    X_test_sel = X_test.copy()

    # Train a model to get feature importance
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), 
                               max_features=n_features, threshold=-np.inf)
    
    # Fit on TRAIN
    selector.fit(X_train_sel, y_train)
    
    # Get the selected feature names
    selected_features = X_train_sel.columns[selector.get_support()]
    
    # Filter columns on both TRAIN and TEST
    X_train_final = X_train_sel[selected_features]
    X_test_final = X_test_sel[selected_features]
    
    # Return y_train unchanged as selection does not change sample size
    return X_train_final, X_test_final, y_train, "ModelSelect_FeatureEng"

def approach_interaction_generation(X_train, X_test, numeric_cols):
    """
    Approach 2: Feature Generation (Interactions and Polynomial features).
    """
    X_train_gen = X_train.copy()
    X_test_gen = X_test.copy()
    
    # Example 1: Interaction feature (multiplication of the first two numeric columns)
    if len(numeric_cols) >= 2:
        col_a, col_b = numeric_cols[0], numeric_cols[1]
        new_col_name = f'{col_a}_x_{col_b}'
        X_train_gen[new_col_name] = X_train_gen[col_a] * X_train_gen[col_b]
        X_test_gen[new_col_name] = X_test_gen[col_a] * X_test_gen[col_b]

    # Example 2: Simple Polynomial feature (square the first numeric column)
    if len(numeric_cols) >= 1:
        col_a = numeric_cols[0]
        new_col_name = f'{col_a}_sq'
        X_train_gen[new_col_name] = X_train_gen[col_a] ** 2
        X_test_gen[new_col_name] = X_test_gen[col_a] ** 2
        
    return X_train_gen, X_test_gen, "Interaction_FeatureEng"