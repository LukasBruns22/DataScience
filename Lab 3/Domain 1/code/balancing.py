import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

def balance_data(X_train, y_train, strategy='smote'):
    """
    Balances the training dataset using Oversampling strategies.

    Parameters:
    -----------
    X_train : pd.DataFrame
        The training features (must be encoded and imputed before SMOTE).
    y_train : pd.Series
        The training target.
    strategy : str, default='smote'
        - 'oversampling': Randomly duplicates samples from the minority class.
        - 'smote': Generates synthetic samples for the minority class using KNN.

    Returns:
    --------
    X_train_res, y_train_res : pd.DataFrame, pd.Series
        The balanced training data.
    """
    
    print(f"--- Balancing Data using Strategy: '{strategy}' ---")
    print(f"Original class distribution:\n{y_train.value_counts()}")

    if strategy == 'oversampling':
        sampler = RandomOverSampler(random_state=42)
        
    elif strategy == 'smote':
        sampler = SMOTE(random_state=42)
        
    else:
        raise ValueError("Strategy must be 'oversampling' or 'smote'.")

    # Apply resampling
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    print(f"New class distribution:\n{y_train_res.value_counts()}")
    
    # Preserve DataFrame column names if input was a DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    
    return X_train_res, y_train_res

# --- Example Usage ---
if __name__ == "__main__":
    # Mock Data (Imbalanced)
    # 90 samples of class 0, 10 samples of class 1
    X_train = pd.DataFrame({'feature1': range(100), 'feature2': range(100)})
    y_train = pd.Series([0]*90 + [1]*10)

    print("--- Test 1: SMOTE ---")
    X_res_smote, y_res_smote = balance_data(X_train, y_train, strategy='smote')
    
    print("\n--- Test 2: Oversampling ---")
    X_res_over, y_res_over = balance_data(X_train, y_train, strategy='oversampling')