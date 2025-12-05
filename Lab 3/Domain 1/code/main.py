import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from data_loader import load_and_split_data
from model_evaluator import train_and_evaluate, compare_strategies_averaged
from encoding_and_mvi import encode_features
from outliers import handle_outliers
from scaling import scale_features
from balancing import balance_data
from feature_selection import select_features

# --- CONFIGURATION ---
FILE_PATH = 'Lab 3/Domain 1/code/traffic_accidents.csv'
TARGET_COL = 'crash_type'
PERFORMANCE_HISTORY = []

def run_step_tournament(step_name, X_train, X_test, y_train, y_test, 
                        func, param_grid, model_type, 
                        baseline_score=-1.0, # NEW ARGUMENT
                        updates_y=False, needs_y_in_func=False):
    """
    Runs a tournament. If the winner is worse than 'baseline_score',
    we REJECT the transformation and return the original data.
    """
    print(f"\n{'-'*60}")
    print(f"TOURNAMENT STEP: {step_name.upper()} for {model_type.upper()}")
    print(f"{'-'*60}")

    step_results = []
    
    # 1. Run all challengers
    for i, params in enumerate(param_grid):
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, str)])
        full_name = f"{step_name}_{config_desc}"
        
        print(f"Testing config: {full_name}")
        
        try:
            # Create copies to protect original data during trial
            # (Though our functions usually copy, this is extra safety)
            if updates_y:
                X_tr_curr, y_tr_curr = func(X_train, y_train, **params)
                X_te_curr, y_te_curr = X_test, y_test
            elif needs_y_in_func:
                X_tr_curr, X_te_curr = func(X_train, X_test, y_train, **params)
                y_tr_curr, y_te_curr = y_train, y_test
            else:
                X_tr_curr, X_te_curr = func(X_train, X_test, **params)
                y_tr_curr, y_te_curr = y_train, y_test

            # Evaluate
            res = train_and_evaluate(X_tr_curr, y_tr_curr, X_te_curr, y_te_curr, model_type, full_name)
            step_results.append(res)
            PERFORMANCE_HISTORY.append(res)
            
        except Exception as e:
            print(f" -> SKIPPED due to error: {e}")

    # 2. Select Winner of this round
    best_approach_name, best_score = compare_strategies_averaged(step_results, metric='f1_score')
    
    # 3. CHECK AGAINST BASELINE (The "Greedy" Logic)
    # We allow the first step (baseline < 0) to always pass because we need encoded data.
    if baseline_score > 0 and best_score < baseline_score:
        print(f"\n>>> 🛑 REJECTION: Best new score ({best_score:.4f}) is worse than baseline ({baseline_score:.4f}).")
        print(f"    Keeping previous dataset (Skipping {step_name}).")
        # Return ORIGINAL data and OLD score
        return X_train, X_test, y_train, y_test, baseline_score

    # 4. Apply Winner (Advance)
    print(f"\n>>> ✅ ACCEPTANCE: New best score ({best_score:.4f}) beats/matches baseline.")
    print(f"    Applying winning transformation ({best_approach_name})...")
    
    best_params = None
    for params in param_grid:
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, str)])
        if f"{step_name}_{config_desc}" == best_approach_name:
            best_params = params
            break

    if updates_y:
        X_tr_win, y_tr_win = func(X_train, y_train, **best_params)
        return X_tr_win, X_test, y_tr_win, y_test, best_score
    elif needs_y_in_func:
        X_tr_win, X_te_win = func(X_train, X_test, y_train, **best_params)
        return X_tr_win, X_te_win, y_train, y_test, best_score
    else:
        X_tr_win, X_te_win = func(X_train, X_test, **best_params)
        return X_tr_win, X_te_win, y_train, y_test, best_score


def run_full_pipeline_for_model(model_name, X_train_init, X_test_init, y_train_init, y_test_init):
    print(f"\n\n{'='*80}")
    print(f"STARTING OPTIMIZATION PIPELINE FOR: {model_name.upper()}")
    print(f"{'='*80}")
    
    X_train, X_test = X_train_init, X_test_init
    y_train, y_test = y_train_init, y_test_init
    
    # Current best score (starts at -1 to force Step 1 acceptance)
    current_best = -1.0
    
    # --- STEP 1: MVI & ENCODING (Mandatory) ---
    grid_step1 = [{'mvi_strategy': 'statistical'}, {'mvi_strategy': 'constant'}]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "1.MVI", X_train, X_test, y_train, y_test,
        func=encode_features, param_grid=grid_step1, model_type=model_name,
        baseline_score=current_best 
    )
    
    initial_valid_score = current_best

    # --- STEP 2: OUTLIERS ---
    grid_step2 = [{'strategy': 'truncate'}, {'strategy': 'replace'}]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "2.Outliers", X_train, X_test, y_train, y_test,
        func=handle_outliers, param_grid=grid_step2, model_type=model_name,
        baseline_score=current_best
    )

    # --- STEP 3: SCALING ---
    grid_step3 = [{'strategy': 'standardization'}, {'strategy': 'normalization'}]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "3.Scaling", X_train, X_test, y_train, y_test,
        func=scale_features, param_grid=grid_step3, model_type=model_name,
        baseline_score=current_best
    )

    # --- STEP 4: BALANCING ---
    grid_step4 = [{'strategy': 'smote'}, {'strategy': 'oversampling'}]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "4.Balancing", X_train, X_test, y_train, y_test,
        func=balance_data, param_grid=grid_step4, model_type=model_name,
        baseline_score=current_best, updates_y=True
    )

    # --- STEP 5: FEATURE SELECTION ---
    grid_step5 = [{'strategy': 'kbest'}, {'strategy': 'correlation'}]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "5.Feat_Sel", X_train, X_test, y_train, y_test,
        func=select_features, param_grid=grid_step5, model_type=model_name,
        baseline_score=current_best, needs_y_in_func=True
    )
    
    print(f"\n>>> FINAL {model_name.upper()} DATA SHAPE: {X_train.shape}")
    print(f">>> FINAL {model_name.upper()} F1 SCORE: {current_best:.4f}")
    
    return initial_valid_score, current_best


def plot_performance_evolution():
    if not PERFORMANCE_HISTORY: return
    df = pd.DataFrame(PERFORMANCE_HISTORY)
    df['Step'] = df['approach'].apply(lambda x: x.split('_')[0])
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='approach', y='f1_score', hue='model', palette='viridis')
    plt.title('Tournament Results: Model Performance Evolution', fontsize=16)
    plt.xticks(rotation=90) 
    plt.ylabel('F1 Score (Weighted)')
    plt.xlabel('Strategy Evaluated')
    plt.legend(title='Model', loc='lower right')
    plt.tight_layout()
    plt.savefig('Lab 3/Domain 1/plots/tournament_results.png')
    print("\nGraphique sauvegardé : Lab 3/Domain 1/plots/tournament_results.png")
    # plt.show() 

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_and_split_data(FILE_PATH, TARGET_COL)

        # 1. KNN PIPELINE
        # Use copies so KNN transformations don't affect NB
        X_tr_knn, X_te_knn = X_train_raw.copy(), X_test_raw.copy()
        y_tr_knn, y_te_knn = y_train_raw.copy(), y_test_raw.copy()
        
        init_knn, final_knn = run_full_pipeline_for_model(
            'knn', X_tr_knn, X_te_knn, y_tr_knn, y_te_knn
        )

        # 2. NB PIPELINE
        X_tr_nb, X_te_nb = X_train_raw.copy(), X_test_raw.copy()
        y_tr_nb, y_te_nb = y_train_raw.copy(), y_test_raw.copy()
        
        init_nb, final_nb = run_full_pipeline_for_model(
            'nb', X_tr_nb, X_te_nb, y_tr_nb, y_te_nb
        )

        print("\n" + "="*60)
        print("PIPELINE OPTIMIZATION COMPLETE")
        print("="*60)
        
        print(f"\nRESULTS SUMMARY:")
        print(f"{'Model':<10} | {'Initial (Step 1)':<15} | {'Final (Step 5)':<15} | {'Improvement':<15}")
        print("-" * 65)
        print(f"{'KNN':<10} | {init_knn:.4f}           | {final_knn:.4f}         | {(final_knn - init_knn):.4f}")
        print(f"{'NB':<10}  | {init_nb:.4f}           | {final_nb:.4f}          | {(final_nb - init_nb):.4f}")
        print("-" * 65)

        plot_performance_evolution()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()