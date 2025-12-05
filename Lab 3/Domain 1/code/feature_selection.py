import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X_train, X_test, y_train, strategy='low_variance', threshold=None, k=20):
    """
    Selects features based on the strategies defined in the lecture:
    1. Low Variance (Data Quality)
    2. Redundancy (Correlation)
    3. Relevance (SelectKBest - likely from your previous slides)
    
    Parameters:
    -----------
    strategy : str
        'low_variance' - Removes features with variance below a threshold.
        'redundancy'   - Removes features highly correlated with others.
        'kbest'        - Selects top K features based on relevance to target.
    threshold : float
        - For 'low_variance': The max variance to drop (e.g., 0.1, 1.0).
        - For 'redundancy': The correlation limit (e.g., 0.9, 0.5).
    """
    X_train_sel = X_train.copy()
    X_test_sel = X_test.copy()
    
    print(f"--- Feature Selection Strategy: '{strategy}' ---")

    # --- STRATEGY 1: Low Variance (Professor's First Step) ---
    if strategy == 'low_variance':
        # Default threshold if none provided (Professor used roughly 1.0 to 3.0 in examples)
        if threshold is None: threshold = 1.0 
        
        # Calculate variance manually to match professor's "std * std" logic
        summary = X_train_sel.describe()
        variances = summary.loc["std"] * summary.loc["std"]
        
        # Identify columns to drop (variance < threshold)
        to_drop = variances[variances < threshold].index.tolist()
        
        print(f"  Threshold: {threshold}")
        print(f"  Dropping {len(to_drop)} low variance features: {to_drop}")
        
        X_train_sel = X_train_sel.drop(columns=to_drop)
        X_test_sel = X_test_sel.drop(columns=to_drop)

    # --- STRATEGY 2: Redundancy / Correlation (Professor's Second Step) ---
    elif strategy == 'redundancy':
        # Default threshold if none provided (Professor suggests 0.9, but found 0.5 better for NB)
        if threshold is None: threshold = 0.9
        
        # Create correlation matrix
        corr_matrix = X_train_sel.corr().abs()
        
        # Select upper triangle to avoid double counting
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
        
        print(f"  Threshold: {threshold}")
        print(f"  Dropping {len(to_drop)} redundant features (Correlation > {threshold}): {to_drop}")
        
        X_train_sel = X_train_sel.drop(columns=to_drop)
        X_test_sel = X_test_sel.drop(columns=to_drop)

    # --- STRATEGY 3: Relevance (Your original KBest) ---
    elif strategy == 'kbest':
        print(f"  Selecting top {k} features using f_classif")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train_sel, y_train)
        
        cols_idxs = selector.get_support(indices=True)
        cols_names = X_train_sel.columns[cols_idxs]
        
        X_train_sel = X_train_sel.iloc[:, cols_idxs]
        X_test_sel = X_test_sel.iloc[:, cols_idxs]
        
    else:
        raise ValueError("Strategy must be 'low_variance', 'redundancy', or 'kbest'")

    return X_train_sel, X_test_sel