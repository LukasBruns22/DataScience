import pandas as pd
import numpy as np
from sklearn.utils import resample
from config import STEP5_OUTPUT_PATH, TARGET_COL, RANDOM_STATE, STEP4_OUTPUT_PATH
from data_utils import load_step_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models
import matplotlib.pyplot as plt

STEP_NAME = "step_5_balancing"

# Undersampling: Reduce majority class to balance with the minority class
def undersample_data(X, y):
    """
    Perform undersampling by reducing the size of the majority class.
    Args:
        X: DataFrame with features
        y: Series with target labels
    Returns:
        X_resampled, y_resampled: Resampled DataFrames with balanced classes
    """
    # Combine X and y to handle sampling together
    df = pd.concat([X, y], axis=1)
    majority_class = df[df[TARGET_COL] == 0]
    minority_class = df[df[TARGET_COL] == 1]

    # Perform undersampling on majority class
    majority_class_downsampled = resample(majority_class, 
                                          replace=False,    # Don't replace to avoid duplication
                                          n_samples=len(minority_class), # Match minority class size
                                          random_state=RANDOM_STATE)

    # Concatenate back into a single dataframe
    df_balanced = pd.concat([majority_class_downsampled, minority_class])

    # Separate X and y
    X_resampled = df_balanced.drop(columns=[TARGET_COL])
    y_resampled = df_balanced[TARGET_COL]
    
    return X_resampled, y_resampled


# Replication: Duplicate minority class samples to balance the dataset
def replicate_data(X, y):
    """
    Perform replication by increasing the size of the minority class.
    Args:
        X: DataFrame with features
        y: Series with target labels
    Returns:
        X_resampled, y_resampled: Resampled DataFrames with balanced classes
    """
    # Combine X and y to handle sampling together
    df = pd.concat([X, y], axis=1)
    majority_class = df[df[TARGET_COL] == 0]
    minority_class = df[df[TARGET_COL] == 1]

    # Perform replication on minority class
    minority_class_upsampled = resample(minority_class, 
                                        replace=True,      # Replicate samples
                                        n_samples=len(majority_class), # Match majority class size
                                        random_state=RANDOM_STATE)

    # Concatenate back into a single dataframe
    df_balanced = pd.concat([majority_class, minority_class_upsampled])

    # Separate X and y
    X_resampled = df_balanced.drop(columns=[TARGET_COL])
    y_resampled = df_balanced[TARGET_COL]
    
    return X_resampled, y_resampled


def main():
    # 1) Load the dataset after encoding, imputation, and outlier handling
    df_raw = load_step_data(STEP4_OUTPUT_PATH)

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

    # Visualize class distribution before balancing
    print("\n[Balancing] Visualizing class distribution before balancing.")
    print(y_train.value_counts())
    print(y_test.value_counts())

    # Balance the dataset using Undersampling
    print("\n[Balancing] Using Undersampling to balance the dataset.")
    X_train_undersampled, y_train_undersampled = undersample_data(X_train, y_train)
    X_test_undersampled, y_test_undersampled = undersample_data(X_test, y_test)

    # Balance the dataset using Replication
    print("\n[Balancing] Using Replication to balance the dataset.")
    X_train_replicated, y_train_replicated = replicate_data(X_train, y_train)
    X_test_replicated, y_test_replicated = replicate_data(X_test, y_test)

    # Train models on Undersampled data
    print("[Balancing] Training on undersampled data.")
    metrics_undersampled, conf_undersampled = train_and_evaluate_models(X_train_undersampled, y_train_undersampled, X_test, y_test)

    # Train models on Replicated data
    print("[Balancing] Training on replicated data.")
    metrics_replicated, conf_replicated = train_and_evaluate_models(X_train_replicated, y_train_replicated, X_test, y_test)

    # Compare metrics and decide best method
    print(f"\n[Balancing] Undersampled Data Metrics:")
    print(metrics_undersampled)

    print(f"\n[Balancing] Replicated Data Metrics:")
    print(metrics_replicated)

    X_full = df_raw.drop(columns=[TARGET_COL])
    y_full = df_raw[TARGET_COL]

    # Save the best approach
    if metrics_undersampled["NaiveBayes"]["f1"] > metrics_replicated["NaiveBayes"]["f1"]:
        print(f"\n[Balancing] Best approach: Undersampling")
        X_full_trans, y_full_trans = undersample_data(X_full, y_full)
    else:
        print(f"\n[Balancing] Best approach: Replication")
        X_full_trans, y_full_trans = replicate_data(X_full, y_full)

    # Save best dataset for the next step
    X_full_df = pd.concat([X_full_trans, y_full_trans], axis=1)

    # Save best dataset for the next step
    save_step_data(X_full_df, STEP5_OUTPUT_PATH)
    print(f"[{STEP_NAME}] Saved best transformed dataset to {STEP5_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
