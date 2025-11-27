import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def split_data(df, target_column, test_size=0.3, random_state=42):
    """
    Splits the DataFrame into training and testing sets for features (X) and target (y).
    """
    if df is None:
        return None, None, None, None

    # Drop rows where the target is missing, if any, and handle potential target issues
    if df[target_column].isnull().any():
        print(f"Warning: Dropping {df[target_column].isnull().sum()} rows with missing target values.")
        df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data successfully split into Training ({len(X_train)} samples) and Test ({len(X_test)} samples).")
    return X_train, X_test, y_train, y_test