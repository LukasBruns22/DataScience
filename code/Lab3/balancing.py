import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def approach_smote_oversampling(X_train, X_test, y_train):
    """
    Approach 1: Synthetic Minority Over-sampling Technique (SMOTE).
    Generates synthetic samples for the minority class.
    """
    smote = SMOTE(random_state=42)
    # Fit and resample on TRAIN data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # X_test remains unchanged
    return X_train_resampled, X_test.copy(), y_train_resampled, "SMOTE_Balancing"

def approach_random_undersampling(X_train, X_test, y_train):
    """
    Approach 2: Random Under-sampling (RUS).
    Randomly removes samples from the majority class.
    """
    rus = RandomUnderSampler(random_state=42)
    # Fit and resample on TRAIN data
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    
    # X_test remains unchanged
    return X_train_resampled, X_test.copy(), y_train_resampled, "RUS_Balancing"