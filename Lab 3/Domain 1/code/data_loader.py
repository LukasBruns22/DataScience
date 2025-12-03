import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_and_split_data(
    file_path: str, 
    target_column: str, 
    test_size: float = 0.3, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads data from a file path and splits it into training and testing sets.

    Args:
        file_path (str): The path to the data file (e.g., 'data.csv').
        target_column (str): The name of the column to be used as the target variable (y).
        test_size (float): The proportion of the data to be used for the test set (default is 0.3).
        random_state (int): Seed for random number generator for reproducibility (default is 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        (X_train, X_test, y_train, y_test)
    """
    print(f"--- Loading data from: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
        raise ValueError(f"File '{file_path}' not found.")
        
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in the data.")
        raise ValueError(f"Target column '{target_column}' missing.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]


    # Encode target variable
    target_mapping = {
        "NO INJURY / DRIVE AWAY": 0,
        "INJURY AND / OR TOW DUE TO CRASH": 1
    }
    y = y.map(target_mapping)
    if y.isna().any():
        print("ATTENTION : Certaines valeurs de la target n'ont pas été trouvées dans le mapping (elles sont devenues NaN).")
    else:
        print("Target encoded successfully.")

    # Replace 'Unknown' placeholders with NaN
    missing_placeholders = [
        "UNKNOWN", "Unknown", "unknown", 
        "UNABLE TO DETERMINE", "NOT APPLICABLE", 
        "nan", "?", "-"
    ]
    
    X = X.replace(missing_placeholders, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y 
    )

    print(f"Data shape (X_train): {X_train.shape}")
    print(f"Data shape (X_test): {X_test.shape}")
    
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
           y_train.reset_index(drop=True), y_test.reset_index(drop=True)
