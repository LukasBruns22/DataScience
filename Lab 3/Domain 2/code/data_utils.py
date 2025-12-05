# data_utils.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TARGET_COL

def load_raw_data():
    from config import RAW_DATA_PATH
    return pd.read_csv(RAW_DATA_PATH)

def load_step_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_step_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def train_test_split_xy(df: pd.DataFrame, test_size=0.2):
    """Split into train/test ensuring we separate X and y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
