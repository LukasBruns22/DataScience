"""
VAR (Vector AutoRegressive)
Multivariate time series forecasting for traffic data
"""
from pandas import read_csv, DataFrame
from statsmodels.tsa.vector_ar.var_model import VAR
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
modules_path = os.path.join(project_root, 'package_lab')
if modules_path not in sys.path:
    sys.path.append(modules_path)
from dslabs_functions import HEIGHT, FORECAST_MEASURES

measure: str = "R2"
TARGET_COL = "Total"

# --- Load multivariate pre-processed data ---
print("Loading multivariate data...")
train = read_csv("Datasets/traffic_forecasting/processed_train_multi.csv", index_col="Datetime", parse_dates=True)
test = read_csv("Datasets/traffic_forecasting/processed_test_multi.csv", index_col="Datetime", parse_dates=True)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")

# Infer frequency
if train.index.freq is None:
    train.index.freq = train.index.inferred_freq
if test.index.freq is None:
    test.index.freq = test.index.inferred_freq

# ==========================================
# VAR STUDY (Simpler, faster)
# ==========================================
def var_study(train: DataFrame, test: DataFrame, measure: str = "R2", target_col: str = "Total"):
    """
    Study VAR model performance with different lag orders.
    """
    print(f"\n{'='*60}")
    print(f"VAR MODEL STUDY")
    print(f"{'='*60}\n")
    
    # Test different lag orders
    p_values = [1, 2, 4, 6, 8, 12]
    
    results = []
    best_model = None
    best_params = {"name": "VAR", "metric": measure, "params": ()}
    best_performance = -100000
    
    fig, ax = plt.subplots(1, 1, figsize=(HEIGHT * 1.5, HEIGHT))
    
    scores = []
    for p in p_values:
        print(f"Testing VAR(p={p})...")
        try:
            # Fit VAR model
            model = VAR(train)
            fitted_model = model.fit(maxlags=p, ic=None)
            
            # Forecast
            forecast_input = train.values[-fitted_model.k_ar:]
            prd_tst = fitted_model.forecast(forecast_input, steps=len(test))
            prd_tst_df = DataFrame(prd_tst, columns=train.columns, index=test.index)
            
            # Evaluate on target column
            eval_score = FORECAST_MEASURES[measure](test[target_col], prd_tst_df[target_col])
            print(f"  VAR(p={p}) -> {measure}={eval_score:.4f}")
            
            scores.append(eval_score)
            
            if eval_score > best_performance:
                best_performance = eval_score
                best_params["params"] = (p,)
                best_model = fitted_model
                
            results.append({
                'p': p,
                'score': eval_score
            })
            
        except Exception as e:
            print(f"  VAR(p={p}) -> FAILED: {e}")
            scores.append(np.nan)
    
    # Plot results
    valid_p = [p_values[i] for i in range(len(scores)) if not np.isnan(scores[i])]
    valid_scores = [s for s in scores if not np.isnan(s)]
    
    ax.plot(valid_p, valid_scores, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Lag Order (p)', fontsize=12)
    ax.set_ylabel(f'{measure} Score', fontsize=12)
    ax.set_title(f'VAR Model Study ({measure})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Lab6/results/VAR/parameters_study.png", dpi=300)
    print(f"\nSaved: Lab6/results/VAR/parameters_study.png")
    
    print(f"\n{'='*60}")
    print(f"BEST VAR MODEL: p={best_params['params'][0]} -> {measure}={best_performance:.4f}")
    print(f"{'='*60}\n")
    
    return best_model, best_params

# ==========================================
# EVALUATION AND PLOTTING
# ==========================================
def plot_multivariate_forecasting_eval(test_true: DataFrame, test_pred: DataFrame, 
                                       title: str = "", target_col: str = "Total"):
    """Plot evaluation metrics for multivariate forecast"""
    
    metrics = {
        'MSE': mean_squared_error(test_true[target_col], test_pred[target_col]),
        'MAE': mean_absolute_error(test_true[target_col], test_pred[target_col]),
        'R2': r2_score(test_true[target_col], test_pred[target_col]),
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # MSE
    axes[0].bar(['Test'], [metrics['MSE']], color='#3498db', alpha=0.7)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Mean Squared Error')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE
    axes[1].bar(['Test'], [metrics['MAE']], color='#2ecc71', alpha=0.7)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R2
    axes[2].bar(['Test'], [metrics['R2']], color='#e74c3c', alpha=0.7)
    axes[2].set_ylabel('R2 Score', fontsize=12)
    axes[2].set_title('R² Score')
    axes[2].set_ylim([max(-1, metrics['R2'] - 0.2), 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return metrics

def plot_multivariate_forecast(train: DataFrame, test: DataFrame, pred: DataFrame,
                               title: str = "", target_col: str = "Total"):
    """Plot forecast vs actual for target column"""
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    # Plot last part of training data for context
    ax.plot(train.index[-200:], train[target_col].iloc[-200:], 
            label='Train', color='#95a5a6', alpha=0.6, linewidth=1)
    
    # Plot actual test data
    ax.plot(test.index, test[target_col], 
            label='Test (Actual)', color='#2ecc71', linewidth=2)
    
    # Plot predictions
    ax.plot(pred.index, pred[target_col], 
            label='Test (Predicted)', color='#e74c3c', 
            linestyle='--', linewidth=2)
    
    # Mark train/test split
    ax.axvline(x=test.index[0], color='black', linestyle=':', 
               linewidth=2, label='Train/Test Split')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Datetime', fontsize=12)
    ax.set_ylabel('Total Traffic Volume', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create output directories
    os.makedirs("Lab6/results/VAR", exist_ok=True)
    
    # ============================================
    # VAR STUDY
    # ============================================
    best_var_model, best_var_params = var_study(train, test, measure=measure, target_col=TARGET_COL)
    
    # Generate VAR predictions
    best_p = best_var_params["params"][0]
    forecast_input = train.values[-best_var_model.k_ar:]
    var_pred = best_var_model.forecast(forecast_input, steps=len(test))
    var_pred_df = DataFrame(var_pred, columns=train.columns, index=test.index)
    
    # Plot VAR evaluation
    var_metrics = plot_multivariate_forecasting_eval(
        test, var_pred_df,
        title=f"VAR(p={best_p}) Evaluation",
        target_col=TARGET_COL
    )
    plt.savefig("Lab6/results/VAR/evals.png", dpi=300)
    print("Saved: Lab6/results/VAR/evals.png")
    
    # Plot VAR forecast
    plot_multivariate_forecast(
        train, test, var_pred_df,
        title=f"VAR(p={best_p}) Forecast",
        target_col=TARGET_COL
    )
    plt.savefig("Lab6/results/VAR/forecast.png", dpi=300)
    print("Saved: Lab6/results/VAR/forecast.png")
    
    # ============================================
    # FINAL RESULTS
    # ============================================
    print(f"\n{'='*60}")
    print("FINAL VAR METRICS")
    print(f"{'='*60}")
    
    print(f"\nVAR(p={best_p}) Metrics:")
    for metric, value in var_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("MULTIVARIATE VAR MODELING COMPLETE")
    print("="*60)
