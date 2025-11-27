import pandas as pd
import numpy as np
from scipy.stats import zscore

def approach_capping_iqr(X_train, X_test, numeric_cols, factor=1.5):
    """
    Approach 1: Capping/Winsorization based on IQR (Interquartile Range).
    """
    X_train_cap = X_train.copy()
    X_test_cap = X_test.copy()
    
    for col in numeric_cols:
        Q1 = X_train_cap[col].quantile(0.25)
        Q3 = X_train_cap[col].quantile(0.75)
        IQR = Q3 - Q1
        
        upper_bound = Q3 + factor * IQR
        lower_bound = Q1 - factor * IQR
        
        # Apply capping (clip) to both train and test sets using bounds from the train set
        X_train_cap[col] = np.clip(X_train_cap[col], lower_bound, upper_bound)
        X_test_cap[col] = np.clip(X_test_cap[col], lower_bound, upper_bound)

    return X_train_cap, X_test_cap, "Capping_Outliers"

def approach_outlier_removal_zscore(X_train, X_test, y_train, numeric_cols, threshold=3):
    """
    Approach 2: Outlier Removal based on Z-score (only on the training set).
    IMPORTANT: This modifies the size of X_train and y_train.
    """
    X_train_rem = X_train.copy()
    y_train_rem = y_train.copy()
    
    # Calculate Z-scores on the training set
    z_scores = X_train_rem[numeric_cols].apply(zscore)
    
    # Identify rows to keep (non-outliers)
    is_not_outlier = (np.abs(z_scores) < threshold).all(axis=1)
    
    X_train_final = X_train_rem[is_not_outlier]
    y_train_final = y_train_rem.loc[X_train_final.index]
    
    # The test set remains unchanged
    return X_train_final, X_test.copy(), y_train_final, "Removal_Outliers"