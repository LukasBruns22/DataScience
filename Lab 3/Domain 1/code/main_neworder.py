import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from data_loader import load_and_split_data
from model_evaluator import train_and_evaluate, compare_strategies_averaged

from encoding_and_mvi import encode_and_impute 
from outliers import handle_outliers
from scaling import scale_features
from balancing import balance_data
from feature_selection import select_features
from feature_generation import generate_features

# --- CONFIGURATION ---
FILE_PATH = 'traffic_accidents_cleaned.csv'
TARGET_COL = 'crash_type'
PERFORMANCE_HISTORY = []


def run_step_tournament(step_name, X_train, X_test, y_train, y_test, 
                        func, param_grid, model_type, 
                        baseline_score=-1.0,
                        updates_y=False, needs_y_in_func=False,
                        include_skip=True):
    
    print(f"\n{'-'*60}")
    print(f"TOURNAMENT STEP: {step_name.upper()} for {model_type.upper()}")
    print(f"{'-'*60}")

    step_results = []
    
    # Helper to safely evaluate mixed data (numbers + strings)
    def safe_evaluate(X_tr, y_tr, X_te, y_te, name):
        # Create a temporary copy that ONLY keeps numbers
        # This prevents KNN from crashing on "SIGNALIZED" strings
        X_tr_safe = X_tr.select_dtypes(include=['number'])
        X_te_safe = X_te.select_dtypes(include=['number'])
        
        # If no numeric columns exist, we can't evaluate
        if X_tr_safe.shape[1] == 0:
            raise ValueError("No numeric columns available for evaluation")
            
        return train_and_evaluate(X_tr_safe, y_tr, X_te_safe, y_te, model_type, name)

    # 1. Test SKIP option
    if include_skip:
        print(f"Testing config: {step_name}_skip")
        try:
            # Use safe_evaluate instead of direct call
            res = safe_evaluate(X_train, y_train, X_test, y_test, f"{step_name}_skip")
            step_results.append(res)
            PERFORMANCE_HISTORY.append(res)
        except Exception as e:
            print(f" -> SKIP option failed: {e}")

    # 2. Test Strategies
    for i, params in enumerate(param_grid):
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, (str, int, float))])
        full_name = f"{step_name}_{config_desc}"
        print(f"Testing config: {full_name}")

        try:
            # Apply the transformation (Generates features, keeps strings)
            if updates_y:
                X_tr_curr, y_tr_curr = func(X_train, y_train, **params)
                X_te_curr, y_te_curr = X_test, y_test
            elif needs_y_in_func:
                X_tr_curr, X_te_curr = func(X_train, X_test, y_train, **params)
                y_tr_curr, y_te_curr = y_train, y_test
            else:
                X_tr_curr, X_te_curr = func(X_train, X_test, **params)
                y_tr_curr, y_te_curr = y_train, y_test

            # Evaluate SAFELY (ignores strings just for scoring)
            res = safe_evaluate(X_tr_curr, y_tr_curr, X_te_curr, y_te_curr, full_name)
            step_results.append(res)
            PERFORMANCE_HISTORY.append(res)

        except Exception as e:
            print(f" -> SKIPPED due to error: {e}")

    # 3. Pick Winner
    best_name, best_score = compare_strategies_averaged(step_results, metric='f1_score')

    # Handle failures
    if best_name is None:
        print(">>> 🛑 CRITICAL: All strategies failed. Returning original data.")
        return X_train, X_test, y_train, y_test, baseline_score

    # Check against baseline
    if include_skip and best_name == f"{step_name}_skip":
        print(f"\n>>> ⏭️  SKIP: Best option was to skip this step (score: {best_score:.4f}).")
        return X_train, X_test, y_train, y_test, best_score

    if baseline_score > 0 and best_score < baseline_score:
        print(f"\n>>> 🛑 REJECTION: Best new score ({best_score:.4f}) is worse than baseline ({baseline_score:.4f}).")
        return X_train, X_test, y_train, y_test, baseline_score

    print(f"\n>>> ✅ ACCEPTANCE: New best score ({best_score:.4f}) accepted.")
    print(f"    Applying best transformation ({best_name})...")

    # 4. Apply Best Transformation PERMANENTLY (Keeping strings for next step!)
    best_params = None
    for params in param_grid:
        config_desc = "_".join([str(v) for v in params.values() if isinstance(v, (str, int, float))])
        if f"{step_name}_{config_desc}" == best_name:
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
    """
    Run the full optimization pipeline with CORRECTED ordering.
    """
    print(f"\n\n{'='*80}")
    print(f"STARTING OPTIMIZATION PIPELINE FOR: {model_name.upper()}")
    print(f"{'='*80}")
    
    X_train, X_test = X_train_init.copy(), X_test_init.copy()
    y_train, y_test = y_train_init.copy(), y_test_init.copy()
    
    current_best = -1.0
    
    # -----------------------------------------------------
    # STEP 1: FEATURE GENERATION
    # -----------------------------------------------------
    grid_step1 = [
        {'strategy': 'simple'}, 
        {'strategy': 'advanced'}
    ]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "1.Feat_Gen", X_train, X_test, y_train, y_test,
        func=generate_features, param_grid=grid_step1, model_type=model_name,
        baseline_score=current_best, include_skip=True
    )
    initial_valid_score = current_best

    # -----------------------------------------------------
    # STEP 2: PREPARATION (MVI + ENCODING)
    # -----------------------------------------------------
    grid_step2 = [
        {'mvi_strategy': 'statistical', 'encoding_strategy': 'onehot'},
        {'mvi_strategy': 'statistical', 'encoding_strategy': 'semantic', 'semantic_mode': 'parents_only'},
        {'mvi_strategy': 'statistical', 'encoding_strategy': 'semantic', 'semantic_mode': 'leaves_only'},
        {'mvi_strategy': 'constant', 'encoding_strategy': 'onehot'},
    ]
    
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "2.Prep", X_train, X_test, y_train, y_test,
        func=encode_and_impute, param_grid=grid_step2, model_type=model_name,
        baseline_score=current_best, include_skip=False
    )

    # -----------------------------------------------------
    # STEP 3: OUTLIERS
    # -----------------------------------------------------
    grid_step3 = [
        {'strategy': 'truncate'}, 
        {'strategy': 'replace'},
        {'strategy': 'drop'}
    ]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "3.Outliers", X_train, X_test, y_train, y_test,
        func=handle_outliers, param_grid=grid_step3, model_type=model_name,
        baseline_score=current_best, 
        include_skip=True,
        updates_y=True,
        needs_y_in_func=True
    )

    # -----------------------------------------------------
    # STEP 4: SCALING
    # -----------------------------------------------------
    grid_step4 = [
        {'strategy': 'standardization'}, 
        {'strategy': 'normalization'}, 
        {'strategy': 'robust'}, 
        {'strategy': 'power'}
    ]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "4.Scaling", X_train, X_test, y_train, y_test,
        func=scale_features, param_grid=grid_step4, model_type=model_name,
        baseline_score=current_best, include_skip=True
    )

    # -----------------------------------------------------
    # STEP 5: BALANCING
    # -----------------------------------------------------
    grid_step5 = [
        {'strategy': 'smote'}, 
        {'strategy': 'oversampling'}, 
        {'strategy': 'undersampling'}
    ]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "5.Balancing", X_train, X_test, y_train, y_test,
        func=balance_data, param_grid=grid_step5, model_type=model_name,
        baseline_score=current_best, updates_y=True, include_skip=True
    )

    # -----------------------------------------------------
    # STEP 6: FEATURE SELECTION
    # -----------------------------------------------------
    grid_step6 = [
        {'strategy': 'low_variance', 'threshold': 1.0},
        
        {'strategy': 'redundancy', 'threshold': 0.5},
        {'strategy': 'redundancy', 'threshold': 0.75}, 
        {'strategy': 'redundancy', 'threshold': 0.90},
        
        {'strategy': 'kbest', 'k': 15},
        {'strategy': 'kbest', 'k': 25},
        {'strategy': 'kbest', 'k': 35},
    ]
    X_train, X_test, y_train, y_test, current_best = run_step_tournament(
        "6.Feat_Sel", X_train, X_test, y_train, y_test,
        func=select_features, param_grid=grid_step6, model_type=model_name,
        baseline_score=current_best, needs_y_in_func=True, include_skip=True
    )

    print(f"\n>>> FINAL {model_name.upper()} DATA SHAPE: {X_train.shape}")
    print(f">>> FINAL {model_name.upper()} F1 SCORE: {current_best:.4f}")
    
    # --- SAVE FINAL DATA TO CSV ---
    print(f"\n💾 Saving final preprocessed datasets for {model_name.upper()}...")
    X_train.to_csv(f'{model_name}_final_X_train.csv', index=False)
    X_test.to_csv(f'{model_name}_final_X_test.csv', index=False)
    y_train.to_csv(f'{model_name}_final_y_train.csv', index=False)
    y_test.to_csv(f'{model_name}_final_y_test.csv', index=False)
    print(f"    -> Saved {model_name}_final_X_train.csv, etc.")

    return initial_valid_score, current_best


def plot_performance_evolution():
    """Create visualization of performance across all steps, including delta improvements."""
    if not PERFORMANCE_HISTORY:
        return
    
    df = pd.DataFrame(PERFORMANCE_HISTORY)
    df['Step'] = df['approach'].apply(lambda x: x.split('_')[0])

    # Best score per step and model
    best_per_step = df.loc[df.groupby(['Step', 'model'])['f1_score'].idxmax()].copy()

    # Sort steps numerically
    best_per_step['Step_num'] = best_per_step['Step'].str.extract(r'^(\d+)').astype(int)
    best_per_step = best_per_step.sort_values(['model', 'Step_num'])

    # Compute Δ improvement
    deltas = []
    for model in best_per_step['model'].unique():
        model_df = best_per_step[best_per_step['model'] == model].sort_values("Step_num")
        prev_score = None
        for _, row in model_df.iterrows():
            if prev_score is None:
                delta = 0 
            else:
                delta = row['f1_score'] - prev_score
            deltas.append({
                'model': model,
                'Step': row['Step'],
                'Step_num': row['Step_num'],
                'delta_f1': delta
            })
            prev_score = row['f1_score']

    deltas_df = pd.DataFrame(deltas)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))

    sns.barplot(data=df, x='approach', y='f1_score', hue='model',
                palette='viridis', ax=axes[0])
    axes[0].set_title('Tournament Results: All Configurations')
    axes[0].tick_params(axis='x', rotation=90)

    sns.lineplot(data=best_per_step, x='Step', y='f1_score', hue='model',
                marker='o', ax=axes[1], linewidth=2.5)
    axes[1].set_title('Best Performance Per Step')

    sns.barplot(data=deltas_df, x='Step', y='delta_f1', hue='model',
                palette='coolwarm', ax=axes[2])
    axes[2].set_title('Improvement Contributed By Each Step (Δ F1)')
    axes[2].axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig('Lab 3/Domain 1/plots/tournament_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graph updated with Δ-improvement chart.")


if __name__ == "__main__":
    try:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_and_split_data(FILE_PATH, TARGET_COL)

        print("\n" + "="*80)
        print("STARTING MULTI-MODEL OPTIMIZATION")
        print("="*80)

        # 1. KNN PIPELINE
        init_knn, final_knn = run_full_pipeline_for_model(
            'knn', X_train_raw, X_test_raw, y_train_raw, y_test_raw
        )

        # 2. NB PIPELINE
        init_nb, final_nb = run_full_pipeline_for_model(
            'nb', X_train_raw, X_test_raw, y_train_raw, y_test_raw
        )

        # Final Summary
        print("\n" + "="*80)
        print("PIPELINE OPTIMIZATION COMPLETE")
        print("="*80)
        
        print(f"\n{'='*70}")
        print(f"{'RESULTS SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"{'Model':<12} | {'Initial':<12} | {'Final':<12} | {'Improvement':<12} | {'% Change':<12}")
        print("-" * 70)
        
        knn_improvement = final_knn - init_knn
        knn_pct = (knn_improvement / init_knn * 100) if init_knn > 0 else 0
        print(f"{'KNN':<12} | {init_knn:<12.4f} | {final_knn:<12.4f} | {knn_improvement:+<12.4f} | {knn_pct:+<11.2f}%")
        
        nb_improvement = final_nb - init_nb
        nb_pct = (nb_improvement / init_nb * 100) if init_nb > 0 else 0
        print(f"{'NB':<12} | {init_nb:<12.4f} | {final_nb:<12.4f} | {nb_improvement:+<12.4f} | {nb_pct:+<11.2f}%")
        
        print("=" * 70)
        
        if final_knn > final_nb:
            winner = "KNN"
            margin = final_knn - final_nb
        else:
            winner = "Naive Bayes"
            margin = final_nb - final_knn
        
        print(f"\n🏆 BEST MODEL: {winner} (margin: {margin:.4f})")

        plot_performance_evolution()
        
        results_df = pd.DataFrame(PERFORMANCE_HISTORY)
        results_df.to_csv('Lab 3/Domain 1/plots/tournament_detailed_results.csv', index=False)
        print("💾 Detailed results saved: Lab 3/Domain 1/plots/tournament_detailed_results.csv")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()