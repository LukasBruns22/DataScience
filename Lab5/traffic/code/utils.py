import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_timeseries(train, test):

    transf = StandardScaler().fit(train)

    train_scaled = transf.transform(train)
    test_scaled = transf.transform(test)

    train_scaled_df = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    test_scaled_df = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)

    return train_scaled_df, test_scaled_df

def series_train_test_split(data, trn_pct: float = 0.90):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train = df_cp.iloc[:trn_size]
    test = df_cp.iloc[trn_size:]
    return train, test

def split_timeseries(
    df,
    target_column: str = 'Total', 
    train_size: float = 0.3, 
):
    """
    Splits the dataset into training and testing sets.
    """
    print(f"--- Splitting data with train size = {train_size} ---")

    timeseries = df[[target_column]]
    
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
    train_copy = train.copy()
    test_copy = test.copy() 
    train_diff = train_copy.diff()
    test_diff = test_copy.diff()
    if derivative == 2:
        train_diff = train_diff.diff()
        test_diff = test_diff.diff()
    train_diff.dropna(inplace=True)
    test_diff.dropna(inplace=True)         
    return train_diff, test_diff

def smoothing_timeseries(train, test, window_size):
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
