import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import STEP4_OUTPUT_PATH, TARGET_COL, RANDOM_STATE, STEP3_OUTPUT_PATH
from data_utils import load_step_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models
import matplotlib.pyplot as plt

STEP_NAME = "step_4_scaling"

# Normalization (Min-Max Scaling)
def normalize_data(df):
    """
    Normalize the data to a range of [0, 1] using Min-Max scaling.
    Args:
        df: DataFrame with numerical columns.
    Returns:
        DataFrame with normalized values.
    """
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Standardization (Z-score Scaling)
def standardize_data(df):
    """
    Standardize the data to have mean 0 and standard deviation 1.
    Args:
        df: DataFrame with numerical columns.
    Returns:
        DataFrame with standardized values.
    """
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def main():
    # 1) Load the dataset after encoding and missing value imputation
    df_raw = load_step_data(STEP3_OUTPUT_PATH)

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

    # Scaling using Normalization (Min-Max)
    print("\n[Scaling] Using Normalization (Min-Max) scaling.")
    X_train_normalized = normalize_data(X_train)
    X_test_normalized = normalize_data(X_test)

    # Scaling using Standardization (Z-score)
    print("\n[Scaling] Using Standardization (Z-score) scaling.")
    X_train_standardized = standardize_data(X_train)
    X_test_standardized = standardize_data(X_test)

    # Train models on Normalized data
    print("[Scaling] Training on normalized data.")
    metrics_normalized, conf_normalized = train_and_evaluate_models(X_train_normalized, y_train, X_test_normalized, y_test)

    # Train models on Standardized data
    print("[Scaling] Training on standardized data.")
    metrics_standardized, conf_standardized = train_and_evaluate_models(X_train_standardized, y_train, X_test_standardized, y_test)

    # Compare metrics and decide best method
    print(f"\n[Scaling] Normalized Data Metrics:")
    print(metrics_normalized)

    print(f"\n[Scaling] Standardized Data Metrics:")
    print(metrics_standardized)


    X_full = df_raw.drop(columns=[TARGET_COL])
    y_full = df_raw[TARGET_COL]

    # Save the best approach
    if metrics_normalized["NaiveBayes"]["f1"] > metrics_standardized["NaiveBayes"]["f1"]:
        print(f"\n[Scaling] Best approach: Normalization (Min-Max)")
        X_full_trans = normalize_data(X_full)
    else:
        print(f"\n[Scaling] Best approach: Standardization (Z-score)")
        X_full_trans = standardize_data(X_full)

    # Save best dataset for the next step
    X_full_df = pd.concat([X_full_trans, y_full], axis=1)

    # Save best dataset for the next step
    save_step_data(X_full_df, STEP4_OUTPUT_PATH)
    print(f"[{STEP_NAME}] Saved best transformed dataset to {STEP4_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
