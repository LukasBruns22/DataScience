import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer

def select_features(X_train, X_test, y_train, strategy='kbest', k=10):
    """
    Selects the most important features.

    Strategies:
    - 'kbest': Selects top K features based on statistical tests (ANOVA/Mutual Info).
               (Corresponds to "Relevance Measures" slide).
    - 'correlation': Removes features that are highly correlated with each other (>0.80).
                     (Corresponds to the Correlation Matrix slide).
    
    Parameters:
    -----------
    k : int
        Number of features to keep (only for 'kbest').
    """
    X_train_sel = X_train.copy()
    X_test_sel = X_test.copy()
    
    print(f"--- Feature Selection Strategy: '{strategy}' ---")

    if strategy == 'kbest':
        selector = SelectKBest(score_func=f_classif, k=k)
        
        selector.fit(X_train_sel, y_train)
        
        # Get columns to keep
        cols_idxs = selector.get_support(indices=True)
        cols_names = X_train_sel.columns[cols_idxs]
        
        print(f"Selected {k} best features: {list(cols_names)}")
        
        X_train_sel = X_train_sel.iloc[:, cols_idxs]
        X_test_sel = X_test_sel.iloc[:, cols_idxs]

    elif strategy == 'correlation':
        # Create correlation matrix
        corr_matrix = X_train_sel.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
        
        print(f"Dropping {len(to_drop)} highly correlated features (Redundancy): {to_drop}")
        
        X_train_sel = X_train_sel.drop(columns=to_drop)
        X_test_sel = X_test_sel.drop(columns=to_drop)

    else:
        raise ValueError("Strategy must be 'kbest' or 'correlation'")

    return X_train_sel, X_test_sel
