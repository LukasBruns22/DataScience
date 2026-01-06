import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_timeseries(train, test):
    """Scale train and test using StandardScaler. Returns scaled data AND the scaler for inverse transform."""
    scaler = StandardScaler().fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)

    train_scaled_df = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    test_scaled_df = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)

    return train_scaled_df, test_scaled_df, scaler

def series_train_test_split(data, trn_pct: float = 0.90):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train = df_cp.iloc[:trn_size]
    test = df_cp.iloc[trn_size:]
    return train, test

def split_timeseries(
    df,
    columns: list = ['Total'], 
    train_size: float = 0.9, 
):
    """
    Splits the dataset into training and testing sets.
    Can handle single column (univariate) or multiple columns (multivariate).
    """
    print(f"--- Splitting data with train size = {train_size} ---")
    print(f"    Columns: {columns}")

    timeseries = df[columns]
    
    train, test = series_train_test_split(timeseries, trn_pct=train_size)
    
    print(f"Data split completed: {len(train)} training samples and {len(test)} testing samples.")
    
    return train, test

def aggregate_timeseries(train, test, freq, method='sum'):
    """
    Create aggregated datasets at different frequencies.
    """
    train_copy = train.copy()
    test_copy = test.copy()

    if freq == 'original':
        return train_copy, test_copy

    if method == 'sum':
        resampled_train = train_copy.resample(freq).sum()
        resampled_test = test_copy.resample(freq).sum()
    elif method == 'mean':
        resampled_train = train_copy.resample(freq).mean()
        resampled_test = test_copy.resample(freq).mean()
    else:
        raise ValueError("Method must be either 'sum' or 'mean'.")
            
    return resampled_train, resampled_test

def differenciation_timeseries(train, test, derivative):
    """
    Apply differentiation. 
    - derivative=0: no differentiation (baseline)
    - derivative=1: first-order diff
    - derivative=2: second-order diff
    - derivative=96: seasonal diff (removes daily pattern, period=96 for 15-min data)
    """
    if derivative == 0:
        return train.copy(), test.copy()
    
    train_copy = train.copy()
    test_copy = test.copy()
    
    if derivative == 96:
        # Seasonal differencing (period = 96 for daily seasonality at 15-min intervals)
        train_diff = train_copy.diff(periods=96)
        test_diff = test_copy.diff(periods=96)
    elif derivative == 1:
        train_diff = train_copy.diff()
        test_diff = test_copy.diff()
    elif derivative == 2:
        train_diff = train_copy.diff().diff()
        test_diff = test_copy.diff().diff()
    else:
        raise ValueError(f"derivative must be 0, 1, 2, or 96. Got {derivative}")
    
    train_diff.dropna(inplace=True)
    test_diff.dropna(inplace=True)         
    return train_diff, test_diff

def smoothing_timeseries(train, test, window_size):
    """Apply smoothing. window_size=0 means no smoothing (baseline)."""
    if window_size == 0:
        return train.copy(), test.copy()
    
    # 1. Smooth Training Data (Standard)
    train_copy = train.copy()    
    train_smooth = train_copy.rolling(window=window_size).mean()
    train_smooth = train_smooth.dropna()
    
    # 2. Smooth Test Data (With History)
    # Grab the last (window_size - 1) points from train
    last_window_of_train = train.iloc[-(window_size - 1):]
    
    # Stick them onto the front of the test set
    test_with_history = pd.concat([last_window_of_train, test])
    
    # Apply rolling mean
    test_smooth_full = test_with_history.rolling(window=window_size).mean()
    
    # Drop the 'history' part we added, keeping only the valid test points
    # Since we added N-1 points, the first valid point is exactly at index [window_size-1],
    # which corresponds to the first actual test timestamp.
    test_smooth = test_smooth_full.iloc[window_size - 1:]
    
    return train_smooth, test_smooth
