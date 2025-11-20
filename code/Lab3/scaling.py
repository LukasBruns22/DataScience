import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def approach_minmax_scaling(X_train, X_test, numeric_cols):
    """
    Approach 1: Normalization (MinMaxScaler). Scales data to [0, 1].
    """
    X_train_minmax = X_train.copy()
    X_test_minmax = X_test.copy()
    
    scaler = MinMaxScaler()
    
    # Fit on TRAIN and transform TRAIN
    X_train_minmax[numeric_cols] = scaler.fit_transform(X_train_minmax[numeric_cols])
    
    # Transform TEST using the TRAIN parameters
    X_test_minmax[numeric_cols] = scaler.transform(X_test_minmax[numeric_cols])

    return X_train_minmax, X_test_minmax, "MinMax_Scaling"

def approach_standard_scaling(X_train, X_test, numeric_cols):
    """
    Approach 2: Standardization (StandardScaler). Scales data to mean=0, std=1.
    """
    X_train_standard = X_train.copy()
    X_test_standard = X_test.copy()
    
    scaler = StandardScaler()
    
    # Fit on TRAIN and transform TRAIN
    X_train_standard[numeric_cols] = scaler.fit_transform(X_train_standard[numeric_cols])
    
    # Transform TEST using the TRAIN parameters
    X_test_standard[numeric_cols] = scaler.transform(X_test_standard[numeric_cols])

    return X_train_standard, X_test_standard, "Standard_Scaling"