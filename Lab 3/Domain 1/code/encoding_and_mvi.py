import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

ordinal_cols = [
    'damage', 
    'intersection_related_i',
    'most_severe_injury',
]

def encode_features(X_train, X_test, encoding_strategy='mixed', mvi_strategy='statistical', ordinal_cols=ordinal_cols):
    """
    Applies Missing Value Imputation (MVI) THEN encodes categorical features.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        The dataframes to process.
    encoding_strategy : str, default='mixed'
        - 'onehot': All categorical columns are one-hot encoded.
        - 'ordinal': All categorical columns are ordinal encoded.
        - 'mixed': Specific columns (ordinal_cols) are ordinal encoded, others one-hot.
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
    
    
    all_cat_cols = X_train_enc.select_dtypes(include=['object', 'category']).columns.tolist()

    if not all_cat_cols:
        print("No categorical columns found after imputation.")
        return X_train_enc, X_test_enc

    print(f"--- Encoding Strategy: {encoding_strategy} ---")

    # --- STRATEGY: MIXED ---
    if encoding_strategy == 'mixed':
        if ordinal_cols is None:
            raise ValueError("You must provide a list of 'ordinal_cols' for the mixed strategy.")
        
        # 1. Apply Ordinal Encoding to the specified ordinal columns
        # We check if columns exist first to avoid errors
        valid_ordinal = [c for c in ordinal_cols if c in X_train_enc.columns]
        
        if valid_ordinal:
            print(f"Ordinal encoding for: {valid_ordinal}")
            ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            X_train_enc[valid_ordinal] = ord_encoder.fit_transform(X_train_enc[valid_ordinal])
            X_test_enc[valid_ordinal] = ord_encoder.transform(X_test_enc[valid_ordinal])

        # 2. Identify the remaining categorical columns (Nominal)
        nominal_cols = [col for col in all_cat_cols if col not in valid_ordinal]
        
        if nominal_cols:
            print(f"One-Hot encoding for: {nominal_cols}")
            oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            oh_encoder.fit(X_train_enc[nominal_cols])
            
            # Transform and create DataFrames
            train_oh = pd.DataFrame(oh_encoder.transform(X_train_enc[nominal_cols]),
                                    columns=oh_encoder.get_feature_names_out(nominal_cols),
                                    index=X_train_enc.index)
            test_oh = pd.DataFrame(oh_encoder.transform(X_test_enc[nominal_cols]),
                                   columns=oh_encoder.get_feature_names_out(nominal_cols),
                                   index=X_test_enc.index)
            
            # Drop original nominal columns and concat the new one-hot columns
            X_train_enc = pd.concat([X_train_enc.drop(columns=nominal_cols), train_oh], axis=1)
            X_test_enc = pd.concat([X_test_enc.drop(columns=nominal_cols), test_oh], axis=1)

    # --- STRATEGY: ONE-HOT (All) ---
    elif encoding_strategy == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_train_enc[all_cat_cols])
        
        train_oh = pd.DataFrame(encoder.transform(X_train_enc[all_cat_cols]),
                                columns=encoder.get_feature_names_out(all_cat_cols),
                                index=X_train_enc.index)
        test_oh = pd.DataFrame(encoder.transform(X_test_enc[all_cat_cols]),
                               columns=encoder.get_feature_names_out(all_cat_cols),
                               index=X_test_enc.index)
        
        X_train_enc = pd.concat([X_train_enc.drop(columns=all_cat_cols), train_oh], axis=1)
        X_test_enc = pd.concat([X_test_enc.drop(columns=all_cat_cols), test_oh], axis=1)

    # --- STRATEGY: ORDINAL (All) ---
    elif encoding_strategy == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_enc[all_cat_cols] = encoder.fit_transform(X_train_enc[all_cat_cols])
        X_test_enc[all_cat_cols] = encoder.transform(X_test_enc[all_cat_cols])

    else:
        raise ValueError("Unknown encoding strategy.")

    return X_train_enc, X_test_enc