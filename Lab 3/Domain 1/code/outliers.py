import pandas as pd
import numpy as np

def handle_outliers(X_train, X_test, strategy='truncate'):
    """
    Handles outliers in numerical features based on the selected strategy.
    
    Strategies (based on the provided slide):
    - 'truncate': Caps values at the lower and upper bounds (Winsorization).
                  (Corresponds to "new max and min").
    - 'replace': Replaces outliers with the Mean of the column.
                 (Corresponds to "New value -> Mean/Median").

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        The dataframes containing features.
    strategy : str, default='truncate'
        The strategy to use ('truncate' or 'replace').

    Returns:
    --------
    X_train_out, X_test_out : pd.DataFrame
        Dataframes with outliers handled.
    """
    
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()

    numerical_cols = X_train_out.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if X_train_out[col].nunique() > 2]

    if len(numerical_cols) == 0:
        print("No continuous numerical columns found for outlier treatment.")
        return X_train_out, X_test_out

    print(f"--- Handling Outliers Strategy: '{strategy}' on columns: {numerical_cols} ---")

    for col in numerical_cols:
        # 2. Calculate Bounds using IQR (Interquartile Range) on TRAIN set only
        # This prevents data leakage.
        Q1 = X_train_out[col].quantile(0.25)
        Q3 = X_train_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Calculate Mean (for the 'replace' strategy)
        col_mean = X_train_out[col].mean()

        # --- STRATEGY 1: TRUNCATE (Capping) ---
        if strategy == 'truncate':
            # Apply to Train
            X_train_out[col] = np.where(X_train_out[col] < lower_bound, lower_bound, X_train_out[col])
            X_train_out[col] = np.where(X_train_out[col] > upper_bound, upper_bound, X_train_out[col])
            
            # Apply to Test (using Train bounds)
            X_test_out[col] = np.where(X_test_out[col] < lower_bound, lower_bound, X_test_out[col])
            X_test_out[col] = np.where(X_test_out[col] > upper_bound, upper_bound, X_test_out[col])

        # --- STRATEGY 2: REPLACE (Mean) ---
        elif strategy == 'replace':
            # Identify outliers
            train_outliers = (X_train_out[col] < lower_bound) | (X_train_out[col] > upper_bound)
            test_outliers = (X_test_out[col] < lower_bound) | (X_test_out[col] > upper_bound)
            
            # Replace with Mean
            if train_outliers.sum() > 0:
                X_train_out.loc[train_outliers, col] = col_mean
            
            if test_outliers.sum() > 0:
                X_test_out.loc[test_outliers, col] = col_mean

        else:
            raise ValueError("Strategy must be 'truncate' or 'replace'.")

    return X_train_out, X_test_out