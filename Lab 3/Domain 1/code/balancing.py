"""
Improved balancing that handles one-hot encoded features correctly.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def identify_continuous_features(X):
    """
    Identify truly continuous features (not binary/one-hot encoded).
    SMOTE should only be applied to these.
    """
    continuous_cols = []
    
    for col in X.columns:
        unique_count = X[col].nunique()
        unique_vals = set(X[col].unique())
        
        # Skip binary features
        if unique_count == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue
        
        # Skip one-hot encoded (only 0 and 1, but we want more than 2 unique)
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            continue
        
        # This is likely a continuous feature
        continuous_cols.append(col)
    
    return continuous_cols

def balance_data(X_train, y_train, strategy='smote'):
    """
    Balance the training data using various strategies.
    
    FIXED: SMOTE now only uses continuous features to avoid creating constant features.
    
    Parameters:
    -----------
    strategy : str
        'smote' - SMOTE on continuous features only
        'oversampling' - Random oversampling (works with any features)
        'undersampling' - Random undersampling
    """
    print(f"--- Balancing Data using Strategy: '{strategy}' ---")
    print(f"Original class distribution:")
    print(y_train.value_counts())
    
    X_train_balanced = X_train.copy()
    y_train_balanced = y_train.copy()
    
    if strategy == 'smote':
        # Identify continuous features
        continuous_cols = identify_continuous_features(X_train)
        
        print(f"Continuous features for SMOTE: {len(continuous_cols)} / {X_train.shape[1]}")
        
        if len(continuous_cols) < 2:
            print("⚠️  WARNING: Not enough continuous features for SMOTE. Using random oversampling instead.")
            strategy = 'oversampling'
        else:
            # Separate continuous and categorical features
            X_continuous = X_train[continuous_cols]
            X_categorical = X_train.drop(columns=continuous_cols)
            
            # Apply SMOTE only to continuous features
            smote = SMOTE(random_state=42, k_neighbors=5)
            
            try:
                X_continuous_resampled, y_train_balanced = smote.fit_resample(X_continuous, y_train)
                
                # For categorical features, duplicate the minority class rows
                # Get indices of resampled data
                original_indices = X_continuous_resampled.index[:len(X_train)]
                synthetic_indices = X_continuous_resampled.index[len(X_train):]
                
                # Find which original rows were duplicated by SMOTE
                # (SMOTE creates new rows, we need to match them to originals for categorical)
                minority_class = y_train.value_counts().idxmin()
                minority_mask = y_train == minority_class
                minority_indices = y_train[minority_mask].index
                
                # Randomly sample from minority class for categorical features
                n_synthetic = len(synthetic_indices)
                sampled_indices = np.random.choice(minority_indices, size=n_synthetic, replace=True)
                
                # Combine original and sampled categorical features
                X_categorical_original = X_categorical.loc[X_train.index]
                X_categorical_synthetic = X_categorical.loc[sampled_indices].reset_index(drop=True)
                X_categorical_synthetic.index = range(len(X_categorical_original), 
                                                     len(X_categorical_original) + len(X_categorical_synthetic))
                
                X_categorical_resampled = pd.concat([X_categorical_original, X_categorical_synthetic])
                
                # Combine continuous and categorical
                X_continuous_resampled = X_continuous_resampled.reset_index(drop=True)
                X_categorical_resampled = X_categorical_resampled.reset_index(drop=True)
                
                X_train_balanced = pd.concat([
                    X_continuous_resampled,
                    X_categorical_resampled
                ], axis=1)
                
                # Ensure column order matches original
                X_train_balanced = X_train_balanced[X_train.columns]
                
            except Exception as e:
                print(f"⚠️  SMOTE failed: {e}")
                print("Falling back to random oversampling...")
                strategy = 'oversampling'
    
    if strategy == 'oversampling':
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)
    
    elif strategy == 'undersampling':
        rus = RandomUnderSampler(random_state=42)
        X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)
    
    print(f"New class distribution:")
    print(y_train_balanced.value_counts())
    
    return X_train_balanced, y_train_balanced