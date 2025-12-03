import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

def encode_features(X_train, X_test, mvi_strategy='statistical'):
    """
    Applies Missing Value Imputation (MVI) THEN encodes categorical features.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        The dataframes to process.
    mvi_strategy : str, default='statistical'
        - 'statistical': Fills numerical with Mean, categorical with Mode.
        - 'constant': Fills numerical with -1, categorical with 'Missing_Data'.
    ordinal_cols : list of str
        List of column names to be treated as ordinal when strategy='mixed'.

    Returns:
    --------
    X_train_enc, X_test_enc : pd.DataFrame
        The processed dataframes (imputed and encoded).
    """
    
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    print(f"--- MVI Strategy: {mvi_strategy} ---")
    
    # Identify types
    num_cols = X_train_enc.select_dtypes(include=['number']).columns
    cat_cols = X_train_enc.select_dtypes(include=['object', 'category']).columns

    # Strategy 1: Statistical (Mean/Mode)
    if mvi_strategy == 'statistical':
        # Numerical -> Mean
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X_train_enc[num_cols] = num_imputer.fit_transform(X_train_enc[num_cols])
            X_test_enc[num_cols] = num_imputer.transform(X_test_enc[num_cols])
        
        # Categorical -> Mode (Most Frequent)
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train_enc[cat_cols] = cat_imputer.fit_transform(X_train_enc[cat_cols])
            X_test_enc[cat_cols] = cat_imputer.transform(X_test_enc[cat_cols])

    # Strategy 2: Constant (Fill Value)
    elif mvi_strategy == 'constant':
        # Numerical -> -1
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='constant', fill_value=-1)
            X_train_enc[num_cols] = num_imputer.fit_transform(X_train_enc[num_cols])
            X_test_enc[num_cols] = num_imputer.transform(X_test_enc[num_cols])
        
        # Categorical -> 'Missing_Data'
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing_Data')
            X_train_enc[cat_cols] = cat_imputer.fit_transform(X_train_enc[cat_cols])
            X_test_enc[cat_cols] = cat_imputer.transform(X_test_enc[cat_cols])
    
    else:
        raise ValueError("mvi_strategy must be 'statistical' or 'constant'")
    
    
    print(f"--- Encoding ---")

    binary_mapping = {'Y': 1, 'N': 0}
    
    col_binary = 'intersection_related_i'
    if col_binary in X_train_enc.columns:
        X_train_enc[col_binary] = X_train_enc[col_binary].map(binary_mapping).fillna(0).astype(int)
        X_test_enc[col_binary] = X_test_enc[col_binary].map(binary_mapping).fillna(0).astype(int)
    
    cyclical_cols = {
        'crash_hour': 23,
        'crash_month': 12,
        'crash_day_of_week': 7
    }

    for col, max_val in cyclical_cols.items():
        if col in X_train_enc.columns:
            X_train_enc[f'{col}_sin'] = np.sin(2 * np.pi * X_train_enc[col] / max_val)
            X_train_enc[f'{col}_cos'] = np.cos(2 * np.pi * X_train_enc[col] / max_val)
            
            X_test_enc[f'{col}_sin'] = np.sin(2 * np.pi * X_test_enc[col] / max_val)
            X_test_enc[f'{col}_cos'] = np.cos(2 * np.pi * X_test_enc[col] / max_val)
            
            X_train_enc.drop(columns=[col], inplace=True)
            X_test_enc.drop(columns=[col], inplace=True)


    damage_order = ['$500 OR LESS', '501 - 1500', 'OVER $1500']
    injury_order = [
        'NO INDICATION OF INJURY', 
        'REPORTED, NOT EVIDENT',
        'NONINCAPACITATING INJURY', 
        'INCAPACITATING INJURY', 
        'FATAL'
    ]
    ordinal_mappings = {
        'damage': damage_order,
        'most_severe_injury': injury_order
    }
    for col, categories_list in ordinal_mappings.items():
        if col in X_train_enc.columns:
            ord_enc = OrdinalEncoder(categories=[categories_list], handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_enc[col] = ord_enc.fit_transform(X_train_enc[[col]])
            X_test_enc[col] = ord_enc.transform(X_test_enc[[col]])
  
    remaining_cat_cols = X_train_enc.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if remaining_cat_cols:
        oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        oh_encoder.fit(X_train_enc[remaining_cat_cols])
        
        # Transform train
        train_oh = pd.DataFrame(
            oh_encoder.transform(X_train_enc[remaining_cat_cols]),
            columns=oh_encoder.get_feature_names_out(remaining_cat_cols),
            index=X_train_enc.index
        )
        # Transform test
        test_oh = pd.DataFrame(
            oh_encoder.transform(X_test_enc[remaining_cat_cols]),
            columns=oh_encoder.get_feature_names_out(remaining_cat_cols),
            index=X_test_enc.index
        )
        
        X_train_enc = pd.concat([X_train_enc.drop(columns=remaining_cat_cols), train_oh], axis=1)
        X_test_enc = pd.concat([X_test_enc.drop(columns=remaining_cat_cols), test_oh], axis=1)

    return X_train_enc, X_test_enc