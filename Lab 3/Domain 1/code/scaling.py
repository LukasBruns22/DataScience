"""
Improved scaling that doesn't scale binary/one-hot encoded features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

def identify_scalable_features(X):
    """
    Identify features that should be scaled (continuous, non-binary).
    """
    scalable_cols = []
    
    for col in X.columns:
        unique_count = X[col].nunique()
        unique_vals = set(X[col].unique())
        
        # Skip binary features (0/1)
        if unique_count == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue
        
        # This is likely a continuous feature that should be scaled
        scalable_cols.append(col)
    
    return scalable_cols

def scale_features(X_train, X_test, strategy='standardization'):
    """
    Scale continuous features while preserving binary/one-hot encoded features.
    
    Parameters:
    -----------
    strategy : str
        'standardization' - StandardScaler (mean=0, std=1)
        'normalization' - MinMaxScaler (range 0-1)
        'robust' - RobustScaler (uses median and IQR, good for outliers)
        'power' - PowerTransformer (makes data more Gaussian)
    """
    # Identify which features should be scaled
    scalable_cols = identify_scalable_features(X_train)
    non_scalable_cols = [col for col in X_train.columns if col not in scalable_cols]
    
    print(f"--- Scaling Strategy: '{strategy}' ---")
    print(f"  Scalable features: {len(scalable_cols)}")
    print(f"  Non-scalable (binary/one-hot): {len(non_scalable_cols)}")
    
    if len(scalable_cols) == 0:
        print("  ⚠️  No features to scale! Returning original data.")
        return X_train.copy(), X_test.copy()
    
    # Separate scalable and non-scalable features
    X_train_scalable = X_train[scalable_cols]
    X_test_scalable = X_test[scalable_cols]
    
    X_train_non_scalable = X_train[non_scalable_cols] if non_scalable_cols else None
    X_test_non_scalable = X_test[non_scalable_cols] if non_scalable_cols else None
    
    # Apply scaling only to scalable features
    if strategy == 'standardization':
        scaler = StandardScaler()
    elif strategy == 'normalization':
        scaler = MinMaxScaler()
    elif strategy == 'robust':
        scaler = RobustScaler()
    elif strategy == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fit and transform
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_scalable),
        columns=scalable_cols,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_scalable),
        columns=scalable_cols,
        index=X_test.index
    )
    
    # Combine scaled and non-scaled features (preserving original column order)
    if non_scalable_cols:
        X_train_final = pd.concat([X_train_scaled, X_train_non_scalable], axis=1)
        X_test_final = pd.concat([X_test_scaled, X_test_non_scalable], axis=1)
        
        # Restore original column order
        X_train_final = X_train_final[X_train.columns]
        X_test_final = X_test_final[X_test.columns]
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    
    print(f"  ✓ Scaling complete")
    
    return X_train_final, X_test_final