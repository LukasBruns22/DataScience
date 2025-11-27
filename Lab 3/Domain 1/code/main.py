import pandas as pd
import numpy as np

from data_loader import load_data, split_data
from model_evaluator import train_and_evaluate_model, compare_and_select_best

from encoding import approach_ohe_encoding, approach_label_encoding
from imputation import approach_mean_mode_imputation, approach_knn_imputation
from outliers import approach_capping_iqr, approach_outlier_removal_zscore
from scaling import approach_minmax_scaling, approach_standard_scaling
from balancing import approach_smote_oversampling, approach_random_undersampling
from feature_selection import approach_model_based_selection, approach_interaction_generation


# --- UTILITY FUNCTION FOR RUNNING ONE STEP ---
def run_kdd_step(X_train, X_test, y_train, y_test, step_name, approaches_list, *args):
    """
    Generic function to run one KDD preparation step, model evaluation, and selection.
    approaches_list: list of tuples (preparation_function, suffix_name, is_y_modifier)
    """
    print(f"\n=======================================================")
    print(f"STARTING KDD STEP: {step_name}")
    print(f"=======================================================")
    
    all_results = []
    
    # Use copies for the approaches to ensure they start from the same base
    X_train_base, X_test_base, y_train_base = X_train.copy(), X_test.copy(), y_train.copy()

    for prep_func, prep_suffix, is_y_modifier in approaches_list:
        
        # --- 2. Apply Preparation ---
        if is_y_modifier:
            # Methods that modify the size of X_train/y_train (Outlier Removal, Balancing)
            # The preparation function must return (X_train_new, X_test_new, y_train_new, name)
            X_train_prep, X_test_prep, y_train_prep, _ = prep_func(X_train_base, X_test_base, y_train_base, *args)
        else:
            # All other methods (Encoding, Imputation, Scaling, Capping, Generation, Selection)
            # The preparation function must return (X_train_new, X_test_new, name)
            X_train_prep, X_test_prep, _ = prep_func(X_train_base, X_test_base, *args)
            y_train_prep = y_train_base.copy() # y_train size is unchanged
            
        # --- 3. Train and Evaluate Models ---
        full_prep_name = f"{prep_suffix}_{step_name}"
        
        # KNN Model
        results_knn = train_and_evaluate_model(X_train_prep, X_test_prep, y_train_prep, y_test, 'KNN', full_prep_name)
        all_results.append(results_knn)

        # Naive Bayes Model
        results_nb = train_and_evaluate_model(X_train_prep, X_test_prep, y_train_prep, y_test, 'NaiveBayes', full_prep_name)
        all_results.append(results_nb)

    # --- 4. Compare and Select the Best ---
    X_train_best, X_test_best, y_train_best, df_results = compare_and_select_best(all_results)
    
    print(f"🎉 Best dataset ({df_results['Preparation'].iloc[0]}) selected. Proceeding to next step.")
    
    # Return the best prepared data and the target for chaining
    return X_train_best, X_test_best, y_train_best, y_test


def full_pipeline(file_path, target, numeric_cols_initial, cat_cols_initial, ord_cols_initial):
    """Orchestrates the entire KDD data preparation process."""
    
    # 0. Initial Load and Split
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df, target)
    
    if X_train is None:
        return

    # --- 1. ENCODING ---
    # We first clean up potential NaNs in categorical columns to prevent encoder errors
    X_train[cat_cols_initial + ord_cols_initial] = X_train[cat_cols_initial + ord_cols_initial].fillna('Missing_Cat')
    X_test[cat_cols_initial + ord_cols_initial] = X_test[cat_cols_initial + ord_cols_initial].fillna('Missing_Cat')

    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'Encoding', [
            (lambda X_tr, X_te, y_tr: approach_ohe_encoding(X_tr, X_te, cat_cols_initial), "OHE", False),
            (lambda X_tr, X_te, y_tr: approach_label_encoding(X_tr, X_te, ord_cols_initial), "LabelE", False)
        ]
    )

    # --- 2. IMPUTATION ---
    # After encoding, nearly all columns should be numeric. We only impute numeric features.
    imputation_numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'Imputation', [
            (lambda X_tr, X_te, y_tr: approach_mean_mode_imputation(X_tr, X_te, imputation_numeric_cols, []), "MeanMode", False),
            (lambda X_tr, X_te, y_tr: approach_knn_imputation(X_tr, X_te, imputation_numeric_cols), "KNN", False)
        ], numeric_cols=imputation_numeric_cols
    )
    
    # --- 3. OUTLIERS TREATMENT ---
    outlier_numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'Outliers', [
            (lambda X_tr, X_te, y_tr: approach_capping_iqr(X_tr, X_te, outlier_numeric_cols), "Capping", False),
            (lambda X_tr, X_te, y_tr: approach_outlier_removal_zscore(X_tr, X_te, y_tr, outlier_numeric_cols), "Removal", True) # True because it modifies y_train size
        ], numeric_cols=outlier_numeric_cols
    )

    # --- 4. SCALING ---
    scaling_numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'Scaling', [
            (lambda X_tr, X_te, y_tr: approach_minmax_scaling(X_tr, X_te, scaling_numeric_cols), "MinMax", False),
            (lambda X_tr, X_te, y_tr: approach_standard_scaling(X_tr, X_te, scaling_numeric_cols), "Standard", False)
        ], numeric_cols=scaling_numeric_cols
    )
    
    # --- 5. BALANCING ---
    # Note: X_train and X_test must be fully numerical here (which they are after scaling)
    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'Balancing', [
            (lambda X_tr, X_te, y_tr: approach_smote_oversampling(X_tr, X_te, y_tr), "SMOTE", True), # True because it modifies y_train size
            (lambda X_tr, X_te, y_tr: approach_random_undersampling(X_tr, X_te, y_tr), "RUS", True)  # True because it modifies y_train size
        ]
    )

    # --- 6. FEATURE SELECTION AND GENERATION ---
    # The models selected the best dataset from the previous step, which is already scaled and balanced.
    
    X_train, X_test, y_train, y_test = run_kdd_step(
        X_train, X_test, y_train, y_test, 'FeatureEng', [
            (lambda X_tr, X_te, y_tr: approach_model_based_selection(X_tr, X_te, y_tr), "Select", True), # True because it uses y_train to select features
            (lambda X_tr, X_te, y_tr: approach_interaction_generation(X_tr, X_te, X_train.columns.tolist()), "Generate", False)
        ]
    )

    print("\n\n🎉 KDD Data Preparation Pipeline Completed!")
    print(f"Final best training set shape: {X_train.shape}")


if __name__ == '__main__':
    # --- CONFIGURATION (Dataset 1) ---
    # **REPLACE THESE WITH YOUR DATASET DETAILS**
    DATASET_FILE_1 = 'data/dataset_1.csv'
    TARGET_COLUMN_1 = 'Survived'           
    NUMERIC_COLS_1 = ['Age', 'Fare']       
    CATEGORICAL_COLS_1 = ['Sex', 'Embarked'] 
    ORDINAL_COLS_1 = ['Pclass']            
    # ---------------------------------
    
    print("\n\n==================== STARTING DATASET 1 ====================")
    full_pipeline(DATASET_FILE_1, TARGET_COLUMN_1, NUMERIC_COLS_1, CATEGORICAL_COLS_1, ORDINAL_COLS_1)

    # --- CONFIGURATION (Dataset 2) ---
    # Uncomment and modify for your second dataset run
    # DATASET_FILE_2 = 'data/dataset_2.csv'
    # TARGET_COLUMN_2 = 'target_col'
    # NUMERIC_COLS_2 = ['feature_A', 'feature_B']
    # CATEGORICAL_COLS_2 = ['feature_C']
    # ORDINAL_COLS_2 = ['feature_D']
    
    # print("\n\n==================== STARTING DATASET 2 ====================")
    # full_pipeline(DATASET_FILE_2, TARGET_COLUMN_2, NUMERIC_COLS_2, CATEGORICAL_COLS_2, ORDINAL_COLS_2)