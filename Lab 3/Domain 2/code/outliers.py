import pandas as pd
import numpy as np
from config import STEP3_OUTPUT_PATH, TARGET_COL, RANDOM_STATE, STEP1_AND_2_OUTPUT_PATH
from data_utils import load_step_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

STEP_NAME = "step_2_outliers_imputing"


# Outlier Imputation: Replace outliers with mean using IQR method
def impute_outliers_iqr(df, iqr_factor=1.5):
    """
    Impute outliers by replacing them with the mean of the column using IQR method.
    Args:
        df: DataFrame with numerical columns.
        iqr_factor: The factor by which the IQR is multiplied to define outliers.
    Returns:
        DataFrame with outliers imputed.
    """

    # Only apply to numerical columns with more than 2 unique values
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if df[col].nunique() > 2]

    print(f"[Imputation] Processing numerical columns: {numerical_cols}")

    if len(numerical_cols) == 0:
        print("[Imputation] No numerical columns with sufficient variance for outlier treatment.")
        return df

    print(f"[Imputation] Number of rows before IQR outlier imputation: {df.shape[0]}")

    df_clean = df.copy()
    
    # Loop through each numerical column to calculate IQR and impute outliers individually
    for col in numerical_cols:
        # Calculate the IQR (Interquartile Range) for the current column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR

        # Identify outliers based on IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Replace outliers with the mean of the column
        col_imputer = SimpleImputer(strategy='mean')
        
        # Apply the imputer to the current column
        df_clean[col] = col_imputer.fit_transform(df_clean[[col]])

        # Replace only the outliers with the imputed values
        df_clean[col] = np.where(outliers, df_clean[col], df[col])

    print(f"[Imputation] Number of rows after IQR outlier imputation: {df_clean.shape[0]}")
    
    return df_clean

# Outlier Truncation: Cap outliers at lower/upper bounds using IQR method
def truncate_outliers_iqr(df, iqr_factor=1.5):
    """
    Truncate outliers by capping them at the lower and upper bounds using IQR method.
    Args:
        df: DataFrame with numerical columns.
        iqr_factor: The factor by which the IQR is multiplied to define outliers.
    Returns:
        DataFrame with outliers truncated (capped).
    """
    # Only apply to numerical columns with more than 2 unique values
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if df[col].nunique() > 2]

    print(f"[Truncation] Processing numerical columns: {numerical_cols}")

    if len(numerical_cols) == 0:
        print("[Truncation] No numerical columns with sufficient variance for outlier treatment.")
        return df

    print(f"[Truncation] Number of rows before IQR outlier truncation: {df.shape[0]}")

    df_clean = df.copy()

    # Loop through each numerical column to calculate IQR and truncate outliers individually
    for col in numerical_cols:
        # Calculate the IQR (Interquartile Range) for the current column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR

        # Identify outliers based on IQR
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Truncate outliers by capping them at the lower and upper bounds
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])

    print(f"[Truncation] Number of rows after IQR outlier truncation: {df_clean.shape[0]}")
    
    return df_clean



def main():
    # 1) Load the dataset after encoding and missing value imputation
    df_raw = load_step_data(STEP1_AND_2_OUTPUT_PATH)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(df_raw)

    N_SAMPLES = 200_000
    if len(X_train) > N_SAMPLES:
        print(f"[Downsampling] Reducing training set from {len(X_train)} → {N_SAMPLES}")
        sample_idx = np.random.choice(X_train.index, size=N_SAMPLES, replace=False)
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
    else:
        print(f"[Downsampling] Training size < {N_SAMPLES}, skipping.")

    # Handle outliers using IQR method (imputation)
    print("\n[Outlier Imputation] Using IQR method for outlier imputation.")
    X_train_imputed = impute_outliers_iqr(X_train, iqr_factor=1.5)
    X_test_imputed = impute_outliers_iqr(X_test, iqr_factor=1.5)

    # Handle outliers using IQR method (removal)
    print("\n[Outlier Removal] Using IQR method for outlier removal.")
    X_train_truncate = truncate_outliers_iqr(X_train, iqr_factor=1.5)
    X_test_truncate = truncate_outliers_iqr(X_test, iqr_factor=1.5)

    # Train models on Imputed data
    print("[Outlier Imputation] Training on IQR imputed data.")
    metrics_imputed, conf_imputed = train_and_evaluate_models(X_train_imputed, y_train, X_test_imputed, y_test)

    # Train models on Truncated data
    print("[Outlier Removal] Training on IQR truncated data.")
    metrics_truncated, conf_truncated = train_and_evaluate_models(X_train_truncate, y_train, X_test_truncate, y_test)

    # Compare metrics and decide best method
    print(f"\n[Outlier Imputation] Imputed Data Metrics:")
    print(metrics_imputed)

    print(f"\n[Outlier Removal] Truncated Data Metrics:")
    print(metrics_truncated)

    X_full = df_raw.drop(columns=[TARGET_COL])
    y_full = df_raw[TARGET_COL]

    # Save the best approach
    if metrics_imputed["NaiveBayes"]["f1"] > metrics_truncated["NaiveBayes"]["f1"]:
        print(f"\n[Outlier Handling] Best approach: Imputation")
        X_full_trans = impute_outliers_iqr(X_full, iqr_factor=1.5)
    else:
        print(f"\n[Outlier Handling] Best approach: Removal")
        X_full_trans = truncate_outliers_iqr(X_full, iqr_factor=1.5)

    # Save best dataset for the next step
    X_full_df = pd.concat([X_full_trans, y_full], axis=1)

    print("Columns at final stage:", X_full_df.columns.tolist())

    save_step_data(X_full_df, STEP3_OUTPUT_PATH)
    print(f"[{STEP_NAME}] Saved best transformed dataset to {STEP3_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
