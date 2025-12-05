"""
Diagnostic script to check if preprocessing steps are actually changing the data.
Run this to identify issues in your pipeline.
"""

import pandas as pd
import numpy as np
from scipy import stats
from data_loader import load_and_split_data
from encoding_and_mvi import encode_and_impute
from outliers import handle_outliers
from scaling import scale_features
from feature_generation import generate_features
from balancing import balance_data

FILE_PATH = 'Lab 3/Domain 1/code/traffic_accidents.csv'
TARGET_COL = 'crash_type'

def compare_datasets(X1, X2, step_name):
    """Compare two datasets to see if they actually changed."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {step_name}")
    print(f"{'='*60}")
    
    # Basic shape check
    print(f"Shape before: {X1.shape}")
    print(f"Shape after:  {X2.shape}")
    
    # Check if data actually changed
    if X1.shape == X2.shape:
        # Compare values (use common columns if shapes differ)
        common_cols = list(set(X1.columns) & set(X2.columns))
        if common_cols:
            X1_common = X1[common_cols]
            X2_common = X2[common_cols]
            
            # Check how many values changed
            differences = (X1_common != X2_common).sum().sum()
            total_values = X1_common.shape[0] * X1_common.shape[1]
            pct_changed = (differences / total_values) * 100
            
            print(f"Values changed: {differences:,} / {total_values:,} ({pct_changed:.2f}%)")
            
            # Statistical comparison
            means_before = X1_common.mean()
            means_after = X2_common.mean()
            stds_before = X1_common.std()
            stds_after = X2_common.std()
            
            mean_change = np.abs(means_before - means_after).mean()
            std_change = np.abs(stds_before - stds_after).mean()
            
            print(f"Average mean change: {mean_change:.6f}")
            print(f"Average std change:  {std_change:.6f}")
            
            if pct_changed < 0.01:
                print("⚠️  WARNING: Less than 0.01% of values changed!")
                return False
        else:
            print("⚠️  No common columns to compare")
            return False
    else:
        print("✓ Shape changed (columns added/removed)")
        return True
    
    return True

def check_outliers_effect(X_train, X_test):
    """Check if outlier handling actually does anything."""
    print(f"\n{'='*60}")
    print(f"OUTLIER DIAGNOSTIC")
    print(f"{'='*60}")
    
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    outliers_found = {}
    for col in numeric_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outlier_count = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
        if outlier_count > 0:
            outliers_found[col] = outlier_count
    
    print(f"\nColumns with outliers: {len(outliers_found)}")
    if outliers_found:
        for col, count in sorted(outliers_found.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / len(X_train)) * 100
            print(f"  {col}: {count} outliers ({pct:.2f}%)")
    else:
        print("  No outliers detected (this might be the problem!)")

def check_scaling_effect(X_train, X_test):
    """Check if scaling actually changes the data."""
    print(f"\n{'='*60}")
    print(f"SCALING DIAGNOSTIC")
    print(f"{'='*60}")
    
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    
    print(f"\nBefore scaling:")
    print(f"  Mean range: [{X_train[numeric_cols].mean().min():.4f}, {X_train[numeric_cols].mean().max():.4f}]")
    print(f"  Std range:  [{X_train[numeric_cols].std().min():.4f}, {X_train[numeric_cols].std().max():.4f}]")
    
    # Try standardization
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, strategy='standardization')
    
    numeric_cols_after = X_train_scaled.select_dtypes(include=['number']).columns
    print(f"\nAfter scaling:")
    print(f"  Mean range: [{X_train_scaled[numeric_cols_after].mean().min():.4f}, {X_train_scaled[numeric_cols_after].mean().max():.4f}]")
    print(f"  Std range:  [{X_train_scaled[numeric_cols_after].std().min():.4f}, {X_train_scaled[numeric_cols_after].std().max():.4f}]")
    
    # Check if binary/one-hot features are being scaled (they shouldn't be)
    binary_cols = [col for col in numeric_cols if X_train[col].nunique() == 2]
    print(f"\n⚠️  Binary columns being scaled: {len(binary_cols)}")
    if binary_cols[:3]:
        print(f"  Examples: {binary_cols[:3]}")

def check_feature_generation_effect(X_train, X_test):
    """Check if feature generation adds anything useful."""
    print(f"\n{'='*60}")
    print(f"FEATURE GENERATION DIAGNOSTIC")
    print(f"{'='*60}")
    
    original_cols = set(X_train.columns)
    
    X_train_gen, X_test_gen = generate_features(X_train, X_test, strategy='advanced')
    
    new_cols = set(X_train_gen.columns) - original_cols
    
    print(f"Original features: {len(original_cols)}")
    print(f"New features: {len(new_cols)}")
    
    if new_cols:
        print(f"\nNew feature names: {list(new_cols)[:10]}")
        
        # Check if new features have variance
        low_variance = []
        for col in new_cols:
            if X_train_gen[col].std() < 0.01:
                low_variance.append(col)
        
        if low_variance:
            print(f"\n⚠️  Low variance features: {len(low_variance)}")
            print(f"  Examples: {low_variance[:5]}")
    else:
        print("⚠️  No new features generated!")

def check_balancing_creates_constants(X_train, y_train):
    """Check if SMOTE is creating constant features."""
    print(f"\n{'='*60}")
    print(f"BALANCING DIAGNOSTIC")
    print(f"{'='*60}")
    
    print(f"Before balancing:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Class distribution:\n{y_train.value_counts()}")
    
    # Check for constant features before
    const_before = (X_train.std() == 0).sum()
    print(f"  Constant features before: {const_before}")
    
    X_train_bal, y_train_bal = balance_data(X_train, y_train, strategy='smote')
    
    print(f"\nAfter balancing:")
    print(f"  Shape: {X_train_bal.shape}")
    print(f"  Class distribution:\n{y_train_bal.value_counts()}")
    
    # Check for constant features after
    const_after = (X_train_bal.std() == 0).sum()
    print(f"  Constant features after: {const_after}")
    
    if const_after > const_before:
        print(f"\n⚠️  CRITICAL: SMOTE created {const_after - const_before} constant features!")
        # Find which features became constant
        const_cols = X_train_bal.columns[X_train_bal.std() == 0]
        print(f"  Constant columns: {list(const_cols)[:10]}")

def check_data_types(X_train):
    """Check if one-hot encoded features are being treated as continuous."""
    print(f"\n{'='*60}")
    print(f"DATA TYPE DIAGNOSTIC")
    print(f"{'='*60}")
    
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    
    # Find binary columns (should be 0/1 only)
    binary_cols = []
    pseudo_binary = []  # Should be binary but has other values
    
    for col in numeric_cols:
        unique_vals = X_train[col].unique()
        if len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
            else:
                pseudo_binary.append(col)
    
    print(f"Binary features (0/1): {len(binary_cols)}")
    print(f"Pseudo-binary features: {len(pseudo_binary)}")
    print(f"Continuous features: {len(numeric_cols) - len(binary_cols) - len(pseudo_binary)}")
    
    # These binary features shouldn't be scaled/transformed
    if binary_cols:
        print(f"\n⚠️  One-hot encoded features that might be getting scaled:")
        print(f"  Count: {len(binary_cols)}")
        print(f"  Examples: {binary_cols[:5]}")

def main():
    print("="*60)
    print("PIPELINE DIAGNOSTICS")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_split_data(FILE_PATH, TARGET_COL)
    
    # 1. Check encoding
    print("\n" + "="*60)
    print("STEP 1: ENCODING & IMPUTATION")
    print("="*60)
    X_train_enc, X_test_enc = encode_and_impute(
        X_train, X_test, 
        mvi_strategy='constant', 
        encoding_strategy='semantic', 
        semantic_mode='leaves_only'
    )
    compare_datasets(X_train, X_train_enc, "After Encoding")
    
    # 2. Check data types
    check_data_types(X_train_enc)
    
    # 3. Check outliers
    check_outliers_effect(X_train_enc, X_test_enc)
    
    # 4. Check outlier handling
    X_train_out, X_test_out = handle_outliers(X_train_enc, X_test_enc, strategy='truncate')
    compare_datasets(X_train_enc, X_train_out, "After Outlier Handling")
    
    # 5. Check scaling
    check_scaling_effect(X_train_out, X_test_out)
    
    # 6. Check feature generation
    check_feature_generation_effect(X_train_out, X_test_out)
    
    # 7. Check balancing
    check_balancing_creates_constants(X_train_out, y_train)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nLook for warnings (⚠️) above to identify issues!")

if __name__ == "__main__":
    main()