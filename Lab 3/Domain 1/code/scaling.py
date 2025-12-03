import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(X_train, X_test, strategy='standardization'):
    """
    Scales numerical features based on the selected strategy.

    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        The training and testing features.
    strategy : str, default='standardization'
        - 'standardization': Uses StandardScaler (z = (x - mean) / std).
                             Result: Mean=0, Std=1.
        - 'normalization': Uses MinMaxScaler.
                           Result: Data bounded between [0, 1].

    Returns:
    --------
    X_train_scaled, X_test_scaled : pd.DataFrame
        The scaled dataframes.
    """
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_cols = X_train_scaled.select_dtypes(include=['int64', 'float64']).columns
    cols_to_scale = [col for col in numerical_cols if X_train_scaled[col].nunique() > 2]

    if not cols_to_scale:
        print("No continuous numerical columns found to scale.")
        return X_train_scaled, X_test_scaled

    print(f"--- Scaling Strategy: '{strategy}' applied to: {cols_to_scale} ---")

    if strategy == 'standardization':
        scaler = StandardScaler()

    elif strategy == 'normalization':
        scaler = MinMaxScaler()
        
    else:
        raise ValueError("Strategy must be 'standardization' or 'normalization'.")

    scaler.fit(X_train_scaled[cols_to_scale])
    
    X_train_scaled[cols_to_scale] = scaler.transform(X_train_scaled[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test_scaled[cols_to_scale])

    return X_train_scaled, X_test_scaled