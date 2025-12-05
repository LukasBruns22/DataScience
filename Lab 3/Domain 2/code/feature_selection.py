import pandas as pd
import numpy as np
from config import STEP5_OUTPUT_PATH, TARGET_COL, RANDOM_STATE, STEP6_OUTPUT_PATH
from data_utils import load_step_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


STEP_NAME = "step_5_feature_selection"


# Select K best features based on statistical tests
def select_kbest_features(X, y, k=10):

    # Use ANOVA F-test
    selector = SelectKBest(f_classif, k=k)

    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features using ANOVA:", selected_features)
    
    # Return only the selected features
    return X[selected_features]


# Remove highly correlated features based on IQR
def select_highly_correlated_features(X, threshold=0.80):
    """
    Remove features that are highly correlated (greater than the threshold).
    Args:
        X: DataFrame with features
        threshold: Correlation threshold to determine highly correlated features
    Returns:
        DataFrame with non-correlated features
    """
    corr_matrix = X.corr().abs()  # Absolute correlation matrix
    
    # Find upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    print(f"Dropping highly correlated features: {to_drop}")

    # Drop the highly correlated features
    X_cleaned = X.drop(columns=to_drop)
    
    return X_cleaned


def main():
    # 1) Load the dataset after encoding, imputation, and outlier handling
    df_raw = load_step_data(STEP5_OUTPUT_PATH)

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

    # Visualize class distribution before feature selection
    print("\n[Feature Selection] Visualizing class distribution before feature selection.")
    print(y_train.value_counts())
    print(y_test.value_counts())

    # Apply Feature Selection using kbest method (ANOVA)
    print("\n[Feature Selection] Using kbest method with ANOVA F-test.")
    X_train_kbest = select_kbest_features(X_train, y_train, k=10)
    X_test_kbest = X_test[X_train_kbest.columns]

    # Apply Feature Selection using Correlation-based method
    print("\n[Feature Selection] Using correlation-based method.")
    X_train_corr = select_highly_correlated_features(X_train, threshold=0.80)
    X_test_corr = X_test[X_train_corr.columns]

    # Train models on kbest (ANOVA) selected data
    print("[Feature Selection] Training on kbest (ANOVA) selected data.")
    metrics_kbest, conf_kbest = train_and_evaluate_models(X_train_kbest, y_train, X_test_kbest, y_test)

    # Train models on correlation-based selected data
    print("[Feature Selection] Training on correlation-based selected data.")
    metrics_corr, conf_corr = train_and_evaluate_models(X_train_corr, y_train, X_test_corr, y_test)

    # Compare metrics and decide best method
    print(f"\n[Feature Selection] kbest (ANOVA) Data Metrics:")
    print(metrics_kbest)

    print(f"\n[Feature Selection] Correlation-based Data Metrics:")
    print(metrics_corr)


    # Save the best approach
    best_method = 'kbest' if metrics_kbest["NaiveBayes"]["f1"] > metrics_corr["NaiveBayes"]["f1"] else 'correlation'
    if best_method == 'kbest':
        print(f"\n[Feature Selection] Best approach: kbest")
        X_full_trans = X_train_kbest
    else:
        print(f"\n[Feature Selection] Best approach: Correlation")
        X_full_trans = X_train_corr

    y_full = y_train

    # Save best dataset for the next step
    X_full_df = pd.concat([X_full_trans, y_full], axis=1)


    # Save best dataset for the next step
    save_step_data(X_full_df, STEP6_OUTPUT_PATH)
    print(f"[{STEP_NAME}] Saved best transformed dataset to {STEP6_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
