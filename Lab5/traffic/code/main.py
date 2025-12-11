import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from utils import smoothing_timeseries, differenciation_timeseries, aggregate_timeseries, split_timeseries, scale_timeseries
from data_loader import load_data
from models import train_and_evaluate, compare_strategies_averaged

PERFORMANCE_HISTORY = []

def run_step_tournament(step_name, train, test, BEST_APPROACH, func, grid, current_baseline_score=np.inf):                      
    
    print(f"\n{'='*60}")
    print(f">>> TOURNAMENT STEP: {step_name.upper()} ({len(params)} Challengers) <<<")
    print(f"{'='*60}")

    step_results = []
    i=0

    if step_name.endswith("Differentiation"):
        print(f"\n[1/2] Testing: 2.Differentiation_First_Derivative")
        train_curr, test_curr = differenciation_timeseries(train, test, lag=1)
        res_persistence = train_and_evaluate(train_curr, test_curr, 'Persistence', f"{step_name}_First_Derivative")
        res_lr  = train_and_evaluate(train_curr, test_curr, 'LR', f"{step_name}_First_Derivative")
        step_results.extend([res_persistence, res_lr])
        PERFORMANCE_HISTORY.extend([res_persistence, res_lr])
        print(f"\n[2/2] Testing: 2.Differentiation_First_Derivative")
        train_curr, test_curr = differenciation_timeseries(train_curr, test_curr, lag=1)
        res_persistence = train_and_evaluate(train_curr, test_curr, 'Persistence', f"{step_name}_Second_Derivative")
        res_lr  = train_and_evaluate(train_curr, test_curr, 'LR', f"{step_name}_Second_Derivative")
        step_results.extend([res_persistence, res_lr])
        PERFORMANCE_HISTORY.extend([res_persistence, res_lr])

    
    for var, params in grid:
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
    for var, params in grid:
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
    grid1 = [
        {'Freq': ['h', 'D', 'W']},
    ]
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "1.Aggregation", train, test, BEST_APPROACH,
        func=aggregate_timeseries,
        param_grid=grid1
    )

    # --- STEP 2: DIFFERENCIATION ---
    grid2 = [
        {'Lag': [1, 4, 96]},
    ]
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "2. Differentiation", train, test, BEST_APPROACH,
        func=differenciation_timeseries,
        param_grid=grid2
    )

    # --- STEP 3: SCALING ---
    grid3 = [
        {'WindowSize': [25, 50, 75, 100]},
    ]
    train, test, current_baseline_score, BEST_APPROACH = run_step_tournament(
        "3.Smoothing", train, test, BEST_APPROACH,
        func=aggregate_timeseries,
        param_grid=grid3
    )

    # END
    print("\n" + "="*60)
    print("PIPELINE OPTIMIZATION COMPLETE")
    print("="*60)
    return train, test, BEST_APPROACH
    

if __name__ == "__main__":
    train, test, BEST_APPROACH = pipeline_optimization()
    print(BEST_APPROACH)