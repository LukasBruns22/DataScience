import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ordinal_cols = [
    'damage', 
    'intersection_related_i', 
    'most_severe_injury'
]

def encode_features(X_train, X_test, strategy='onehot', ordinal_cols=ordinal_cols):
    """
    Encodes categorical features using One-Hot, Ordinal, or a Mixed strategy.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        The dataframes to encode.
    strategy : str
        - 'onehot': All categorical columns are one-hot encoded.
        - 'ordinal': All categorical columns are ordinal encoded.
        - 'mixed': Specific columns (ordinal_cols) are ordinal encoded, 
                   the rest are one-hot encoded.
    ordinal_cols : list of str, optional
        List of column names to be treated as ordinal when strategy='mixed'.

    Returns:
    --------
    X_train_enc, X_test_enc : pd.DataFrame
        The encoded dataframes.
    """
    
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    # Identify all categorical columns automatically
    all_cat_cols = X_train_enc.select_dtypes(include=['object', 'category']).columns.tolist()

    if not all_cat_cols:
        print("No categorical columns found.")
        return X_train_enc, X_test_enc

    print(f"--- Encoding Strategy: {strategy} ---")

    # --- STRATEGY: MIXED ---
    if strategy == 'mixed':
        if ordinal_cols is None:
            raise ValueError("You must provide a list of 'ordinal_cols' for the mixed strategy.")
        
        # 1. Apply Ordinal Encoding to the specified ordinal columns
        print(f"Ordinal encoding for: {ordinal_cols}")
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        X_train_enc[ordinal_cols] = ord_encoder.fit_transform(X_train_enc[ordinal_cols])
        X_test_enc[ordinal_cols] = ord_encoder.transform(X_test_enc[ordinal_cols])

        # 2. Identify the remaining categorical columns (Nominal)
        nominal_cols = [col for col in all_cat_cols if col not in ordinal_cols]
        
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
            # Note: We keep the ordinal columns (already transformed in place)
            X_train_enc = pd.concat([X_train_enc.drop(columns=nominal_cols), train_oh], axis=1)
            X_test_enc = pd.concat([X_test_enc.drop(columns=nominal_cols), test_oh], axis=1)

    # --- STRATEGY: ONE-HOT (All) ---
    elif strategy == 'onehot':
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
    elif strategy == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_enc[all_cat_cols] = encoder.fit_transform(X_train_enc[all_cat_cols])
        X_test_enc[all_cat_cols] = encoder.transform(X_test_enc[all_cat_cols])

    else:
        raise ValueError("Unknown strategy.")

    return X_train_enc, X_test_enc