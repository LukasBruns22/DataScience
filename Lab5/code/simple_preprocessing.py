"""
SARIMA Preprocessing Pipeline
- Load raw traffic data
- Build datetime index
- Split into train/test (90/10)
- Scale univariate target ('Total') for SARIMA
- Save processed datasets and scaler
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DATA_PATH = "Datasets/TrafficTwoMonth.csv"
TARGET_COL = "Total"
TRAIN_RATIO = 0.9

def load_raw_data():
    """Load raw CSV and build datetime index"""
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Build Month column for datetime
    df['Month'] = None
    df.loc[:2111, 'Month'] = '2023-12-'
    df.loc[2112:5087, 'Month'] = '2024-01-'
    df.loc[5088:, 'Month'] = '2024-02-'
    
    # Combine into datetime string and convert
    df['Datetime_Str'] = df['Month'] + df['Date'].astype(str).str.zfill(2) + ' ' + df['Time']
    df['Datetime'] = pd.to_datetime(df['Datetime_Str'], format='%Y-%m-%d %I:%M:%S %p')
    
    df.drop(columns=['Month', 'Datetime_Str'], inplace=True)
    df = df.set_index('Datetime').sort_index()
    
    return df[[TARGET_COL]]

def split_train_test(df, train_ratio=TRAIN_RATIO):
    """Split dataframe into train/test"""
    train_size = int(len(df) * train_ratio)
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()
    return train, test

def scale_data(train, test):
    """Scale target column using StandardScaler"""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns
    )
    return train_scaled, test_scaled, scaler

def save_processed_data(train, test, scaler, output_dir="Datasets/traffic_forecasting"):
    """Save processed CSVs and scaler"""
    os.makedirs(output_dir, exist_ok=True)
    train.index.name = 'Datetime'
    test.index.name = 'Datetime'
    
    train.to_csv(f"{output_dir}/processed_train.csv")
    test.to_csv(f"{output_dir}/processed_test.csv")
    joblib.dump(scaler, f"{output_dir}/scaler_sarima.pkl")
    
    print(f"Saved processed data to {output_dir}")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

if __name__ == "__main__":
    df = load_raw_data()
    train, test = split_train_test(df)
    train_scaled, test_scaled, scaler = scale_data(train, test)
    save_processed_data(train_scaled, test_scaled, scaler)
