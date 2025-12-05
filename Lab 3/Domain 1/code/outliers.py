import pandas as pd
import numpy as np

def handle_outliers(X_train, X_test, y_train=None, strategy='truncate'):
    """
    Handles outliers using strategies from the lecture.
    
    Parameters:
    - y_train: Required ONLY for 'drop' strategy to keep labels aligned.
    """
    
    # Create copies to avoid SettingWithCopy warnings
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    y_train_out = y_train.copy() if y_train is not None else None

    # Identify numeric columns
    # We select all numeric columns, but you can filter specific ones if needed
    numerical_cols = X_train_out.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude likely ID columns or binary columns if any slipped through
    numerical_cols = [c for c in numerical_cols if X_train_out[c].nunique() > 2]

    if not numerical_cols:
        return X_train_out, X_test_out

    print(f"--- Outliers: '{strategy}' on {len(numerical_cols)} columns ---")

    # Dictionary to store bounds so we apply the SAME bounds to Test as Train
    bounds = {}

    for col in numerical_cols:
        # 1. Calculate IQR
        Q1 = X_train_out[col].quantile(0.25)
        Q3 = X_train_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 2. Determine Bounds
        # FIX: If IQR is 0 (sparse data like 'injuries_fatal'), use Standard Deviation instead
        if IQR == 0:
            mean_val = X_train_out[col].mean()
            std_val = X_train_out[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
        bounds[col] = (lower_bound, upper_bound)

    # --- EXECUTE STRATEGY ---

    if strategy == 'drop':
        if y_train_out is None:
            raise ValueError("You must pass 'y_train' to handle_outliers when using strategy='drop'")
            
        # Identify rows to drop (indices)
        rows_to_drop = set()
        for col in numerical_cols:
            lb, ub = bounds[col]
            outliers = X_train_out[(X_train_out[col] < lb) | (X_train_out[col] > ub)].index
            rows_to_drop.update(outliers)
            
        print(f"  Dropping {len(rows_to_drop)} rows from Training set.")
        
        X_train_out = X_train_out.drop(index=list(rows_to_drop))
        y_train_out = y_train_out.drop(index=list(rows_to_drop))
        
        # Note: We do NOT drop from X_test. We usually truncate X_test to match train bounds
        # or leave it alone. Here, we'll apply truncation to Test for consistency.
        for col in numerical_cols:
            lb, ub = bounds[col]
            X_test_out[col] = np.where(X_test_out[col] < lb, lb, X_test_out[col])
            X_test_out[col] = np.where(X_test_out[col] > ub, ub, X_test_out[col])
            
        return X_train_out, X_test_out, y_train_out

    elif strategy == 'truncate':
        for col in numerical_cols:
            lb, ub = bounds[col]
            X_train_out[col] = np.where(X_train_out[col] < lb, lb, X_train_out[col])
            X_train_out[col] = np.where(X_train_out[col] > ub, ub, X_train_out[col])
            
            X_test_out[col] = np.where(X_test_out[col] < lb, lb, X_test_out[col])
            X_test_out[col] = np.where(X_test_out[col] > ub, ub, X_test_out[col])
        
        if y_train is not None:
             return X_train_out, X_test_out, y_train_out
        return X_train_out, X_test_out

    elif strategy == 'replace':
        for col in numerical_cols:
            lb, ub = bounds[col]
            # PROFESSOR FIX: Use Median, not Mean
            median_val = X_train_out[col].median()
            
            train_outliers = (X_train_out[col] < lb) | (X_train_out[col] > ub)
            test_outliers = (X_test_out[col] < lb) | (X_test_out[col] > ub)
            
            if train_outliers.sum() > 0:
                X_train_out.loc[train_outliers, col] = median_val
            if test_outliers.sum() > 0:
                X_test_out.loc[test_outliers, col] = median_val

        if y_train is not None:
             return X_train_out, X_test_out, y_train_out
        return X_train_out, X_test_out
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")