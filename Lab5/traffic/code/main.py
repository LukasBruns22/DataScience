import numpy as np
from utils import smoothing_timeseries, differenciation_timeseries, aggregate_timeseries, split_timeseries, scale_timeseries
from data_loader import load_data
from models import train_and_evaluate, compare_strategies_averaged

PERFORMANCE_HISTORY = []

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
    print(f"\n>>> 🏆 WINNER of {step_name}: {best_approach_name}")
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
    Main function to run the pipeline optimization.
    """
    df = load_data()
    train, test = split_timeseries(df, target_column='Total', train_size=0.9)
    train, test = scale_timeseries(train, test) 

    BEST_APPROACH = []     

    # --- STEP 1: AGGREGATION ---
    grid1 = {'Freq': ['original', 'h', 'D', 'W']}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "1_Aggregation", train, test, BEST_APPROACH,
        func=aggregate_timeseries,
        grid=grid1
    )

    # --- STEP 2: DIFFERENCIATION ---
    grid2 = {'derivative': [1, 2]}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "2_Differentiation", train, test, BEST_APPROACH,
        func=differenciation_timeseries,
        grid=grid2,
        current_baseline_score=current_baseline_score
    )

    # --- STEP 3: SCALING ---
    grid3 = {'WindowSize': [25, 50, 75, 100]}
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "3_Smoothing", train, test, BEST_APPROACH,
        func=smoothing_timeseries,
        grid=grid3,
        current_baseline_score=current_baseline_score
    )

    # END
    print("\n" + "="*60)
    print("PIPELINE OPTIMIZATION COMPLETE")
    print("="*60)
    return train, test, BEST_APPROACH
    

if __name__ == "__main__":
    train, test, BEST_APPROACH = pipeline_optimization()
    print(f"Final Pipeline Strategy: {BEST_APPROACH}")

    # --- ADD THIS CODE TO SAVE YOUR PROCESSED DATA ---
    print("\nSaving processed datasets...")
    
    # Ensure indices have a name for clean CSV saving
    train.index.name = 'Datetime' 
    test.index.name = 'Datetime'

    # Save to CSV
    train.to_csv("processed_train.csv")
    test.to_csv("processed_test.csv")
    
    print("✅ Success! Data saved to 'processed_train.csv' and 'processed_test.csv'.")