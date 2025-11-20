import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

def approach_mean_mode_imputation(X_train, X_test, numeric_cols, categorical_cols):
    """
    Approach 1: Imputation using Mean (numeric) and Mode (categorical).
    """
    X_train_mmi = X_train.copy()
    X_test_mmi = X_test.copy()
    
    # Impute Numeric (Mean)
    if numeric_cols:
        mean_imputer = SimpleImputer(strategy='mean')
        X_train_mmi[numeric_cols] = mean_imputer.fit_transform(X_train_mmi[numeric_cols])
        X_test_mmi[numeric_cols] = mean_imputer.transform(X_test_mmi[numeric_cols])

    # Impute Categorical (Mode)
    if categorical_cols:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        X_train_mmi[categorical_cols] = mode_imputer.fit_transform(X_train_mmi[categorical_cols])
        X_test_mmi[categorical_cols] = mode_imputer.transform(X_test_mmi[categorical_cols])

    return X_train_mmi, X_test_mmi, "MeanMode_Imputation"

def approach_knn_imputation(X_train, X_test):
    """
    Approach 2: Imputation using K-Nearest Neighbors (KNNImputer).
    Data are encoded during the previous step, so we only have numeric columns here.
    """
    X_train_knn = X_train.copy()
    X_test_knn = X_test.copy()

    # Initialize KNNImputer (k=5 is common)
    knn_imputer = KNNImputer(n_neighbors=5)
    
    # Fit on TRAIN and transform TRAIN
    X_train_imputed_array = knn_imputer.fit_transform(X_train_knn)
    X_train_imputed_df = pd.DataFrame(
        X_train_imputed_array, 
        columns=X_train_knn.columns, 
        index=X_train_knn.index
    )
    X_train_knn = X_train_imputed_df
    
    # Transform TEST using the fitted imputer
    X_test_imputed_array = knn_imputer.transform(X_test_knn)
    X_test_imputed_df = pd.DataFrame(
        X_test_imputed_array, 
        columns=X_test_knn.columns, 
        index=X_test_knn.index
    )
    X_test_knn = X_test_imputed_df
    
    return X_train_knn, X_test_knn, "KNN_Imputation"