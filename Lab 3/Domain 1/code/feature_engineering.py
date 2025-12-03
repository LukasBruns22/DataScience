import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer

def generate_features(X_train, X_test, strategy='time_mapping'):
    """
    Generates features. 
    
    Strategies:
    - 'time_mapping': Creates Season, Weekend, Rush Hour features.
    - 'algebraic': Creates interactions by multiplying numerical columns.
                   (Corresponds to 'Algebraic' template in slide).
    """
    X_train_gen = X_train.copy()
    X_test_gen = X_test.copy()
    
    print(f"--- Feature Generation Strategy: '{strategy}' ---")

    if strategy == 'time_mapping':
        print("-> Mapping Weekends (6 & 7)...")
        for df in [X_train_gen, X_test_gen]:
            if 'crash_day_of_week' in df.columns:
                df['is_weekend'] = df['crash_day_of_week'].isin([6, 7]).astype(int)

        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        print("-> Mapping Seasons...")
        for df in [X_train_gen, X_test_gen]:
            if 'crash_month' in df.columns:
                df['season'] = df['crash_month'].map(season_map)

        def get_time_slot(h):
            if 6 <= h <= 9: return 'Morning_Rush'
            elif 16 <= h <= 19: return 'Evening_Rush'
            elif 22 <= h or h <= 5: return 'Night'
            else: return 'Day'

        print("-> Mapping Time Slots...")
        for df in [X_train_gen, X_test_gen]:
            if 'crash_hour' in df.columns:
                df['time_slot'] = df['crash_hour'].apply(get_time_slot)

    elif strategy == 'algebraic':
        nums = X_train_gen.select_dtypes(include=['number']).columns
        cols_to_interact = [c for c in nums if X_train_gen[c].nunique() > 2]
        
        if len(cols_to_interact) >= 2:
            print(f"-> creating interactions for: {cols_to_interact}")
            
            # degree=2 means we create A*B, A^2, B^2. 
            # interaction_only=True means we ONLY create A*B (no squares).
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            # Fit on Train
            new_feats_train = poly.fit_transform(X_train_gen[cols_to_interact])
            new_feats_test = poly.transform(X_test_gen[cols_to_interact])
            
            # Generate clean names
            feat_names = poly.get_feature_names_out(cols_to_interact)
            
            # Create temporary DFs
            df_poly_tr = pd.DataFrame(new_feats_train, columns=feat_names, index=X_train_gen.index)
            df_poly_te = pd.DataFrame(new_feats_test, columns=feat_names, index=X_test_gen.index)
            
            # Keep only the NEW interaction columns (columns containing " ")
            # PolynomialFeatures returns original cols too ("a", "b", "a b"). We want only "a b".
            interaction_cols = [c for c in feat_names if " " in c]
            
            # Add to main DF
            X_train_gen = pd.concat([X_train_gen, df_poly_tr[interaction_cols]], axis=1)
            X_test_gen = pd.concat([X_test_gen, df_poly_te[interaction_cols]], axis=1)
            
            print(f"Created interaction features: {interaction_cols}")
        else:
            print("Not enough numerical columns found for interaction.")

    else:
        raise ValueError("Strategy must be 'time_mapping' or 'algebraic'")

    return X_train_gen, X_test_gen


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
