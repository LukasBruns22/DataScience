import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def approach_ohe_encoding(X_train, X_test):
    """
    Approach 1: One-Hot Encoding (OHE)
    """
    X_train_ohe = X_train.copy()
    X_test_ohe = X_test.copy()

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit on TRAIN and transform TRAIN
    ohe.fit(X_train_ohe)
    
    feature_names = ohe.get_feature_names_out(X_train_ohe.columns)
    encoded_train = ohe.transform(X_train_ohe)
    encoded_train_df = pd.DataFrame(encoded_train, index=X_train_ohe.index, columns=feature_names)
    
    X_train_ohe = X_train_ohe.drop(columns=X_train_ohe.columns).join(encoded_train_df)
    
    # Transform TEST using the fitted encoder
    encoded_test = ohe.transform(X_test_ohe)
    encoded_test_df = pd.DataFrame(encoded_test, index=X_test_ohe.index, columns=feature_names)
    
    X_test_ohe = X_test_ohe.drop(columns=X_test_ohe.columns).join(encoded_test_df)

    return X_train_ohe, X_test_ohe, "OHE_Encoding"


def approach_label_encoding(X_train, X_test):
    """
    Approach 2: Label Encoding 
    """
    X_train_le = X_train.copy()
    X_test_le = X_test.copy()

    for col in X_train_le.columns:
        le = LabelEncoder()
        
        # Fit on TRAIN
        le.fit(X_train_le[col])
        
        # Transform TRAIN
        X_train_le[col] = le.transform(X_train_le[col])
        
        # Transform TEST (handling unseen labels by mapping them to the lowest encoded value, 0)
        unseen_mask = ~X_test_le[col].isin(le.classes_)
        if unseen_mask.any():
            X_test_le.loc[unseen_mask, col] = le.classes_[0]
        
        X_test_le[col] = le.transform(X_test_le[col])

    return X_train_le, X_test_le, "LabelE_Encoding"