import pandas as pd
import numpy as np
from config import STEP6_OUTPUT_PATH, TARGET_COL, RANDOM_STATE, STEP7_OUTPUT_PATH
from data_utils import load_step_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models


STEP_NAME = "step_6_feature_generation"


# Feature Generation Method
def generate_features(X):
    """
    Feature Generation:
    Args:
        X: DataFrame with features
    Returns:
        DataFrame with generated features
    """
    X_generated = X.copy()
    
    # TODO: Add feature generation logic here
    
    print(f"[Feature Generation] Generated {X_generated.shape[1] - X.shape[1]} new features")
    
    return X_generated


def main():
    # 1) Load the dataset after feature selection
    df_raw = load_step_data(STEP6_OUTPUT_PATH)

    # Print current columns before feature generation
    print("\n[Feature Generation] Current columns:")
    print(df_raw.columns.tolist())
    print(f"Total columns: {len(df_raw.columns)}")

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_xy(df_raw)

    print(f"dimensions of training data: {X_train.shape}, test data: {X_test.shape}")
    print(f"dimensions of training labels: {y_train.shape}, test labels: {y_test.shape}")   

    N_SAMPLES = 200_000
    if len(X_train) > N_SAMPLES:
        print(f"[Downsampling] Reducing training set from {len(X_train)} → {N_SAMPLES}")
        sample_idx = np.random.choice(X_train.index, size=N_SAMPLES, replace=False)
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
    else:
        print(f"[Downsampling] Training size < {N_SAMPLES}, skipping.")

    # Visualize class distribution before feature generation
    print("\n[Feature Generation] Visualizing class distribution before feature generation.")
    print(y_train.value_counts())
    print(y_test.value_counts())

    # Apply Feature Generation
    print("\n[Feature Generation] Generating features.")
    X_train_generated = generate_features(X_train)
    X_test_generated = generate_features(X_test)

    # Train models on generated data
    print("[Feature Generation] Training models on generated data.")
    metrics, conf = train_and_evaluate_models(X_train_generated, y_train, X_test_generated, y_test)

    print(f"\n[Feature Generation] Metrics:")
    print(metrics)

    # Combine train and test with their original target values
    X_full_generated = pd.concat([X_train_generated, X_test_generated], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    # Save best dataset for the next step
    X_full_df = pd.concat([X_full_generated, y_full], axis=1)

    save_step_data(X_full_df, STEP7_OUTPUT_PATH)
    print(f"[{STEP_NAME}] Saved transformed dataset to {STEP7_OUTPUT_PATH}")


if __name__ == "__main__":
    main()