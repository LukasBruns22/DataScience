import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_aggregated_datasets(train, test, freq, method='sum'):
    """
    Create aggregated datasets at different frequencies.
    """
    train_copy = train.copy()
    test_copy = test.copy()

    if method == 'sum':
        resampled_train = train_copy.resample(freq).sum()
        resampled_test = test_copy.resample(freq).sum()
    elif method == 'mean':
        resampled_train = train_copy.resample(freq).mean()
        resampled_test = test_copy.resample(freq).mean()
    else:
        raise ValueError("Method must be either 'sum' or 'mean'.")
            
    return resampled_train, resampled_test