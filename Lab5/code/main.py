import numpy as np
import os
from utils import smoothing_timeseries, differenciation_timeseries, aggregate_timeseries, split_timeseries, scale_timeseries
from data_loader import load_data
from models import train_and_evaluate, compare_strategies_averaged

PERFORMANCE_HISTORY = []

# Configuration
UNIVARIATE_COLS = ['Total']
MULTIVARIATE_COLS = ['Total', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']

def run_step_tournament(step_name, train, test, BEST_APPROACH, func, grid, current_baseline_score=np.inf):                      
    
    print(f"\n{'='*60}")
    print(f">>> TOURNAMENT STEP: {step_name.upper()} <<<")
    print(f"{'='*60}")

    step_results = []
    i=0
    
    for var, params in grid.items():
        for param in params:
            full_name = f"{step_name}_{var}_{param}"
            print(f"\n[{i+1}/{len(params)}] Testing: {full_name}")
            
            train_curr, test_curr = func(train, test, param)

            res_persistence = train_and_evaluate(train_curr, test_curr, 'Persistence', full_name)
            res_lr  = train_and_evaluate(train_curr, test_curr, 'LR', full_name)
                
            step_results.extend([res_persistence, res_lr])
            PERFORMANCE_HISTORY.extend([res_persistence, res_lr])
            i += 1

    best_approach_name, best_score, baseline = compare_strategies_averaged(step_results, current_baseline_score)
    current_baseline_score = best_score
    print(f"\n>>> WINNER of {step_name}: {best_approach_name}")
    BEST_APPROACH.append(best_approach_name)

    if baseline:
        print(f"Retaining current dataset without changes for next step.")
        return train, test, current_baseline_score
    
    best_param = None
    for var, params in grid.items():
        for param in params:
            if f"{step_name}_{var}_{param}" == best_approach_name:
                best_param = param
                break
            
    print(f"Applying winning transformation ({best_approach_name}) to pass to next step...")
    train_curr, test_curr = func(train, test, best_param)

    return train_curr, test_curr, current_baseline_score, BEST_APPROACH

def pipeline_optimization():
    """
    Main function to run the pipeline optimization on Total column only.
    Returns the best parameters found.
    """
    df = load_data()
    train, test = split_timeseries(df, columns=UNIVARIATE_COLS, train_size=0.9)
    train, test, scaler = scale_timeseries(train, test)

    BEST_APPROACH = []     

    # --- STEP 1: AGGREGATION ---
    grid1 = {'Freq': ['original', 'h', 'D']}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "1_Aggregation", train, test, BEST_APPROACH,
        func=aggregate_timeseries,
        grid=grid1
    )

    # --- STEP 2: DIFFERENCIATION ---
    grid2 = {'derivative': [0, 1, 2]}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "2_Differentiation", train, test, BEST_APPROACH,
        func=differenciation_timeseries,
        grid=grid2,
        current_baseline_score=current_baseline_score
    )

    # --- STEP 3: SMOOTHING ---
    grid3 = {'WindowSize': [0, 25, 50]}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "3_Smoothing", train, test, BEST_APPROACH,
        func=smoothing_timeseries,
        grid=grid3,
        current_baseline_score=current_baseline_score
    )

    print("\n" + "="*60)
    print("PIPELINE OPTIMIZATION COMPLETE")
    print("="*60)
    return train, test, BEST_APPROACH, scaler

def extract_best_params(best_approach_list):
    """Extract the actual parameter values from best approach names."""
    params = {}
    for approach in best_approach_list:
        if 'Freq' in approach:
            params['freq'] = approach.split('_')[-1]
        elif 'derivative' in approach:
            params['derivative'] = int(approach.split('_')[-1])
        elif 'WindowSize' in approach:
            params['window_size'] = int(approach.split('_')[-1])
    return params

def apply_transformations(train, test, params):
    """Apply the same transformations found during optimization."""
    # Aggregation
    train, test = aggregate_timeseries(train, test, params.get('freq', 'original'))
    # Differentiation
    train, test = differenciation_timeseries(train, test, params.get('derivative', 0))
    # Smoothing
    train, test = smoothing_timeseries(train, test, params.get('window_size', 0))
    return train, test
    

if __name__ == "__main__":
    import joblib
    import json
    
    # Ensure output directories exist
    os.makedirs("Datasets/traffic_forecasting", exist_ok=True)
    os.makedirs("Lab5/outputs", exist_ok=True)
    
    # ============================================
    # RUN OPTIMIZATION ON TOTAL COLUMN ONLY
    # ============================================
    print("\n" + "#"*60)
    print("  OPTIMIZING PIPELINE (on Total column)")
    print("#"*60)
    
    train_uni, test_uni, BEST_APPROACH, scaler_uni = pipeline_optimization()
    print(f"\nBest Strategy: {BEST_APPROACH}")
    
    # Extract best parameters
    best_params = extract_best_params(BEST_APPROACH)
    print(f"Best Parameters: {best_params}")
    
    # Save univariate data
    train_uni.index.name = 'Datetime' 
    test_uni.index.name = 'Datetime'
    train_uni.to_csv("Datasets/traffic_forecasting/processed_train.csv")
    test_uni.to_csv("Datasets/traffic_forecasting/processed_test.csv")
    
    # ============================================
    # APPLY SAME TRANSFORMATIONS TO MULTIVARIATE
    # ============================================
    print("\n" + "#"*60)
    print("  APPLYING BEST TRANSFORMS TO MULTIVARIATE DATA")
    print("#"*60)
    
    df = load_data()
    train_multi, test_multi = split_timeseries(df, columns=MULTIVARIATE_COLS, train_size=0.9)
    train_multi, test_multi, scaler_multi = scale_timeseries(train_multi, test_multi)
    train_multi, test_multi = apply_transformations(train_multi, test_multi, best_params)
    
    print(f"Multivariate shape: train={train_multi.shape}, test={test_multi.shape}")
    
    # Save multivariate data
    train_multi.index.name = 'Datetime' 
    test_multi.index.name = 'Datetime'
    train_multi.to_csv("Datasets/traffic_forecasting/processed_train_multi.csv")
    test_multi.to_csv("Datasets/traffic_forecasting/processed_test_multi.csv")
    
    # ============================================
    # SAVE SCALERS AND PIPELINE INFO
    # ============================================
    joblib.dump(scaler_uni, 'Lab5/outputs/scaler_uni.pkl')
    joblib.dump(scaler_multi, 'Lab5/outputs/scaler_multi.pkl')
    
    pipeline_info = {
        'columns_univariate': UNIVARIATE_COLS,
        'columns_multivariate': MULTIVARIATE_COLS,
        'best_params': best_params,
        'best_approach': BEST_APPROACH,
    }
    with open('Lab5/outputs/pipeline_info.json', 'w') as f:
        json.dump(pipeline_info, f, indent=2)
    
    print("\n" + "="*60)
    print("Success! Saved:")
    print("="*60)
    print("UNIVARIATE:")
    print("   - Datasets/traffic_forecasting/processed_train.csv")
    print("   - Datasets/traffic_forecasting/processed_test.csv")
    print("MULTIVARIATE:")
    print("   - Datasets/traffic_forecasting/processed_train_multi.csv")
    print("   - Datasets/traffic_forecasting/processed_test_multi.csv")
    print("METADATA:")
    print("   - Lab5/outputs/scaler_uni.pkl")
    print("   - Lab5/outputs/scaler_multi.pkl")
    print("   - Lab5/outputs/pipeline_info.json")