import pandas as pd
import numpy as np
from pipeline import run_step_comparison

def generate_traffic_features(X_train, y_train, X_test, y_test):
    """
    Generate MINIMAL high-quality interaction features
    Focus only on interactions that should clearly matter for injury severity
    """
    print("      (Generating high-quality interaction features...)")
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    features_created = 0
    
    # ===== ONLY CREATE STRONG, CLEAR INTERACTIONS =====
    
    # 1. High Risk Cause × Number of Units
    # More vehicles + dangerous cause = compound risk
    if 'prim_contributory_cause_enc' in X_train.columns and 'num_units' in X_train.columns:
        X_train_new['risk_x_units'] = X_train['prim_contributory_cause_enc'] * X_train['num_units']
        X_test_new['risk_x_units'] = X_test['prim_contributory_cause_enc'] * X_test['num_units']
        features_created += 1
    
    # 2. Pedestrian/Cyclist crashes × Intersection
    # Pedestrian crashes at intersections are especially severe
    ped_cols = [c for c in X_train.columns if 'first_crash_type' in c.lower() and 
                ('pedestrian' in c.lower() or 'pedalcyclist' in c.lower())]
    if ped_cols and 'intersection_related_i' in X_train.columns:
        for col in ped_cols:
            col_name = 'ped' if 'pedestrian' in col.lower() else 'cycle'
            X_train_new[f'{col_name}_intersection'] = X_train[col] * X_train['intersection_related_i']
            X_test_new[f'{col_name}_intersection'] = X_test[col] * X_test['intersection_related_i']
            features_created += 1
    
    # 3. High Risk Cause × Intersection
    # Dangerous driver behavior at intersections compounds risk
    if 'prim_contributory_cause_enc' in X_train.columns and 'intersection_related_i' in X_train.columns:
        X_train_new['risk_intersection'] = X_train['prim_contributory_cause_enc'] * X_train['intersection_related_i']
        X_test_new['risk_intersection'] = X_test['prim_contributory_cause_enc'] * X_test['intersection_related_i']
        features_created += 1
    
    # 4. Poor Surface × Poor Lighting (ONLY if columns exist)
    # Can't see bad road conditions = dangerous
    wet_cols = [c for c in X_train.columns if 'roadway_surface' in c.lower() and 'wet' in c.lower()]
    dark_cols = [c for c in X_train.columns if 'lighting' in c.lower() and 'dark' in c.lower()]
    if wet_cols and dark_cols:
        X_train_new['wet_dark'] = X_train[wet_cols[0]] * X_train[dark_cols[0]]
        X_test_new['wet_dark'] = X_test[wet_cols[0]] * X_test[dark_cols[0]]
        features_created += 1
    
    # 5. Multi-vehicle rear-ends (common in congestion)
    rear_cols = [c for c in X_train.columns if 'first_crash_type' in c.lower() and 'rear' in c.lower()]
    if rear_cols and 'num_units' in X_train.columns:
        X_train_new['rear_end_multi'] = X_train[rear_cols[0]] * X_train['num_units']
        X_test_new['rear_end_multi'] = X_test[rear_cols[0]] * X_test['num_units']
        features_created += 1
    
    if features_created == 0:
        print(f"      -> Warning: No interaction features could be created (missing columns)")
    else:
        print(f"      -> Created {features_created} targeted interaction features")
    
    return X_train_new, y_train, X_test_new, y_test


def no_feature_generation(X_train, y_train, X_test, y_test):
    """
    Baseline: No feature generation
    """
    print("      (No feature generation - baseline)")
    return X_train.copy(), y_train, X_test.copy(), y_test


def run_feature_generation_step(X_train, y_train, X_test, y_test):
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=no_feature_generation,
        app2_func=generate_traffic_features,
        step_name="FEATURE GENERATION"
    )