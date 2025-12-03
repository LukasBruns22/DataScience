import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_and_split_data
from model_evaluator import train_and_evaluate, compare_strategies_averaged
from encoding_and_mvi import encode_features
from outliers import handle_outliers
from scaling import scale_features
from balancing import balance_data
from feature_engineering import feature_gen_and_sel 

# --- CONFIGURATION ---
FILE_PATH = 'traffic_crashes.csv'
TARGET_COL = 'crash_type'

PERFORMANCE_HISTORY = []

def run_step_tournament(step_name, X_train, X_test, y_train, y_test, 
                        func, param_grid, updates_y=False, needs_y_in_func=False):
    print(f"\n{'='*60}")
    print(f">>> TOURNAMENT STEP: {step_name.upper()} ({len(param_grid)} Challengers) <<<")
    print(f"{'='*60}")

    step_results = []
    
    for i, params in enumerate(param_grid):
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, str)])
        full_name = f"{step_name}_{config_desc}"
        
        print(f"\n[{i+1}/{len(param_grid)}] Testing: {full_name}")
        
        try:
            if updates_y:
                X_tr_curr, y_tr_curr = func(X_train, y_train, **params)
                X_te_curr, y_te_curr = X_test, y_test
            elif needs_y_in_func:
                X_tr_curr, X_te_curr, y_tr_curr, y_te_curr = func(X_train, X_test, y_train, y_test, **params)
            else:
                # Ex: Scaling, Outliers, Imputation
                X_tr_curr, X_te_curr = func(X_train, X_test, **params)
                y_tr_curr, y_te_curr = y_train, y_test

            # Evaluation
            res_knn = train_and_evaluate(X_tr_curr, y_tr_curr, X_te_curr, y_te_curr, 'knn', full_name)
            res_nb  = train_and_evaluate(X_tr_curr, y_tr_curr, X_te_curr, y_te_curr, 'nb', full_name)
            
            step_results.extend([res_knn, res_nb])
            PERFORMANCE_HISTORY.extend([res_knn, res_nb])
            
        except Exception as e:
            print(f"   -> SKIPPED due to error: {e}")


    best_approach_name = compare_strategies_averaged(step_results, metric='f1_score')
    
    print(f"\n>>> 🏆 WINNER of {step_name}: {best_approach_name}")

    
    best_params = None
    for params in param_grid:
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, str)])
        if f"{step_name}_{config_desc}" == best_approach_name:
            best_params = params
            break
            
    print(f"Applying winning transformation ({best_approach_name}) to pass to next step...")
    
    if updates_y:
        X_tr_win, y_tr_win = func(X_train, y_train, **best_params)
        return X_tr_win, X_test, y_tr_win, y_test
    elif needs_y_in_func:
        X_tr_win, X_te_win, y_tr_win, y_te_win = func(X_train, X_test, y_train, y_test, **best_params)
        return X_tr_win, X_te_win, y_tr_win, y_te_win
    else:
        X_tr_win, X_te_win = func(X_train, X_test, **best_params)
        return X_tr_win, X_te_win, y_train, y_test


def plot_performance_evolution():
    if not PERFORMANCE_HISTORY: return
    df = pd.DataFrame(PERFORMANCE_HISTORY)
    df['Step'] = df['approach'].apply(lambda x: x.split('_')[0])
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='approach', y='f1_score', hue='model', palette='viridis')
    plt.title('Tournament Results: Model Performance Evolution', fontsize=16)
    plt.xticks(rotation=90) # Rotation 90 car il y aura beaucoup de noms
    plt.ylabel('F1 Score (Weighted)')
    plt.xlabel('Strategy Evaluated')
    plt.legend(title='Model', loc='lower right')
    plt.tight_layout()
    plt.savefig('tournament_results.png')
    print("\nGraphique sauvegardé : tournament_results.png")
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # 0. LOAD & PRE-CLEAN
        X_train, X_test, y_train, y_test = load_and_split_data(FILE_PATH, TARGET_COL)

        # --- STEP 1: MVI & ENCODING ---
        grid_step1 = [
            {'mvi_strategy': 'statistical', 'encoding_strategy': 'mixed'},
            {'mvi_strategy': 'constant',    'encoding_strategy': 'mixed'},
            {'mvi_strategy': 'statistical', 'encoding_strategy': 'onehot'},
            {'mvi_strategy': 'constant',    'encoding_strategy': 'onehot'}
        ]
        X_train, X_test, y_train, y_test = run_step_tournament(
            "1.MVI_Enc", X_train, X_test, y_train, y_test,
            func=encode_features,
            param_grid=grid_step1
        )

        # --- STEP 2: OUTLIERS ---
        grid_step2 = [
            {'strategy': 'truncate'},
            {'strategy': 'replace'}
        ]
        X_train, X_test, y_train, y_test = run_step_tournament(
            "2.Outliers", X_train, X_test, y_train, y_test,
            func=handle_outliers,
            param_grid=grid_step2
        )

        # --- STEP 3: SCALING ---
        grid_step3 = [
            {'strategy': 'standardization'},
            {'strategy': 'normalization'}
        ]
        X_train, X_test, y_train, y_test = run_step_tournament(
            "3.Scaling", X_train, X_test, y_train, y_test,
            func=scale_features,
            param_grid=grid_step3
        )

        # --- STEP 4: BALANCIN ---
        grid_step4 = [
            {'strategy': 'smote'},
            {'strategy': 'oversampling'}
        ]
        X_train, X_test, y_train, y_test = run_step_tournament(
            "4.Balancing", X_train, X_test, y_train, y_test,
            func=balance_data,
            param_grid=grid_step4,
            updates_y=True
        )

        # --- STEP 5: FEATURE GEN & SEL (4 possibilités) ---
        grid_step5 = [
            {'gen_strategy': 'time_mapping', 'sel_strategy': 'kbest'},
            {'gen_strategy': 'time_mapping', 'sel_strategy': 'correlation'},
            {'gen_strategy': 'algebraic',    'sel_strategy': 'kbest'},
            {'gen_strategy': 'algebraic',    'sel_strategy': 'correlation'}
        ]
        X_train, X_test, y_train, y_test = run_step_tournament(
            "5.Feat_GenSel", X_train, X_test, y_train, y_test,
            func=feature_gen_and_sel, 
            param_grid=grid_step5,
            needs_y_in_func=True 
        )

        # END
        print("\n" + "="*60)
        print("PIPELINE OPTIMIZATION COMPLETE")
        print("="*60)
        plot_performance_evolution()
        print(f"Final Dataset Shape: {X_train.shape}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()