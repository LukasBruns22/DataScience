import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TRAIN_PATH = "datasets/traffic_forecasting/processed_train.csv"
TEST_PATH = "datasets/traffic_forecasting/processed_test.csv"
TARGET = "Total"
SEASONALITY = 96  # Daily cycle (15-min intervals * 24h * 4)

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data():
    print(f"Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=['Datetime'], index_col='Datetime')
    test = pd.read_csv(TEST_PATH, parse_dates=['Datetime'], index_col='Datetime')
    
    # Infer frequency for Statsmodels
    if train.index.freq is None:
        train.index.freq = pd.infer_freq(train.index)
    if test.index.freq is None:
        test.index.freq = pd.infer_freq(test.index)
        
    return train[TARGET], test[TARGET]

# ==========================================
# 2. ARIMA HYPERPARAMETER STUDY
# ==========================================
def run_arima_study(train, test):
    print(f"\n{'='*40}\nSTARTING ARIMA STUDY\n{'='*40}")
    
    # Grid
    p_values = [1, 2, 4, 6]
    d_values = [0, 1, 2]
    q_values = [0, 1, 2]
    
    results = []
    best_score = -float('inf')
    best_cfg = None
    
    total = len(p_values) * len(d_values) * len(q_values)
    curr = 0
    
    # Plotting Data
    plot_data = {d: {q: [] for q in q_values} for d in d_values}
    
    for d in d_values:
        for q in q_values:
            for p in p_values:
                curr += 1
                order = (p, d, q)
                print(f"\r[ARIMA {curr}/{total}] Testing {order}...", end="")
                
                try:
                    model = ARIMA(train, order=order).fit()
                    preds = model.forecast(steps=len(test))
                    r2 = r2_score(test, preds)
                    
                    # Save for plot
                    plot_data[d][q].append((p, r2))
                    
                    if r2 > best_score:
                        best_score = r2
                        best_cfg = order
                        
                except:
                    plot_data[d][q].append((p, np.nan))
                    
    print(f"\nBest ARIMA: {best_cfg} (R2: {best_score:.4f})")
    
    # --- PLOT 1: ARIMA STUDY ---
    fig, axes = plt.subplots(1, len(d_values), figsize=(18, 5), sharey=True)
    if len(d_values) == 1: axes = [axes]
    
    for i, d in enumerate(d_values):
        ax = axes[i]
        for q, values in plot_data[d].items():
            valid_p = [v[0] for v in values if not np.isnan(v[1])]
            valid_r2 = [v[1] for v in values if not np.isnan(v[1])]
            if valid_p:
                ax.plot(valid_p, valid_r2, marker='o', label=f'q={q}')
        
        ax.set_title(f"ARIMA (d={d})")
        ax.set_xlabel("p parameter")
        if i == 0: ax.set_ylabel("R2 Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("ARIMA Hyperparameter Study")
    plt.tight_layout()
    plt.savefig("study_arima.png", dpi=300)
    print("Saved: study_arima.png")
    
    return best_cfg

# ==========================================
# 3. SARIMA HYPERPARAMETER STUDY
# ==========================================
def run_sarima_study(train, test):
    print(f"\n{'='*40}\nSTARTING SARIMA STUDY (s={SEASONALITY})\n{'='*40}")
    
    # Simplified Grid for Speed
    orders = [(1, 0, 1)]
    seasonal_D = [0]
    seasonal_P = [0, 1]
    seasonal_Q = [0, 1] 
    
    results = []
    
    total = len(orders) * len(seasonal_D) * len(seasonal_P) * len(seasonal_Q)
    curr = 0
    
    for order in orders:
        for D in seasonal_D:
            for P in seasonal_P:
                for Q in seasonal_Q:
                    curr += 1
                    s_order = (P, D, Q, SEASONALITY)
                    name = f"SARIMA{order}x{s_order}"
                    print(f"[{curr}/{total}] {name}...", end=" ")
                    
                    try:
                        # Fast approximate fit
                        model = ARIMA(train, order=order, seasonal_order=s_order, 
                                     enforce_stationarity=False).fit(low_memory=True)
                        preds = model.forecast(steps=len(test))
                        r2 = r2_score(test, preds)
                        print(f"R2={r2:.4f}")
                        
                        results.append({
                            "params": (order, s_order),
                            "D": D,
                            "P": P,
                            "R2": r2
                        })
                    except Exception as e:
                        print(f"Failed.")

    # Find Best
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['R2'].idxmax()]
    best_cfg = best_row['params']
    
    print(f"\nBest SARIMA: {best_cfg[0]}x{best_cfg[1]} (R2: {best_row['R2']:.4f})")

    # --- PLOT 2: SARIMA STUDY ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="D", y="R2", hue="P", palette="viridis")
    plt.title("SARIMA Study: Impact of Seasonal Parameters")
    plt.xlabel("Seasonal Difference (D)")
    plt.ylabel("R2 Score")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("study_sarima.png", dpi=300)
    print("Saved: study_sarima.png")
    
    return best_cfg

# ==========================================
# 4. FINAL COMPARISON
# ==========================================
def compare_and_forecast(train, test, arima_cfg, sarima_cfg):
    print(f"\n{'='*40}\nFINAL COMPARISON\n{'='*40}")
    
    # 1. Re-Evaluate Best ARIMA
    print(f"Evaluating Best ARIMA {arima_cfg}...")
    m_arima = ARIMA(train, order=arima_cfg).fit()
    p_arima = m_arima.forecast(steps=len(test))
    
    arima_metrics = {
        "Model": "ARIMA",
        "MSE": mean_squared_error(test, p_arima),
        "MAE": mean_absolute_error(test, p_arima),
        "R2": r2_score(test, p_arima)
    }
    
    # 2. Re-Evaluate Best SARIMA
    print(f"Evaluating Best SARIMA {sarima_cfg[0]}x{sarima_cfg[1]}...")
    m_sarima = ARIMA(train, order=sarima_cfg[0], seasonal_order=sarima_cfg[1]).fit(low_memory=True)
    p_sarima = m_sarima.forecast(steps=len(test))
    
    sarima_metrics = {
        "Model": "SARIMA",
        "MSE": mean_squared_error(test, p_sarima),
        "MAE": mean_absolute_error(test, p_sarima),
        "R2": r2_score(test, p_sarima)
    }
    
    # --- PLOT 3: METRICS COMPARISON ---
    metrics_df = pd.DataFrame([arima_metrics, sarima_metrics])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.barplot(data=metrics_df, x="Model", y="MSE", ax=axes[0], palette="Blues")
    axes[0].set_title("MSE (Lower is Better)")
    
    sns.barplot(data=metrics_df, x="Model", y="MAE", ax=axes[1], palette="Greens")
    axes[1].set_title("MAE (Lower is Better)")
    
    sns.barplot(data=metrics_df, x="Model", y="R2", ax=axes[2], palette="Reds")
    axes[2].set_title("R2 Score (Higher is Better)")
    
    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=300)
    print("Saved: comparison_metrics.png")
    
    # --- PLOT 4: WINNER FORECAST ---
    # Determine Winner (using R2)
    if arima_metrics["R2"] > sarima_metrics["R2"]:
        winner_name = "ARIMA"
        winner_preds = p_arima
        winner_r2 = arima_metrics["R2"]
    else:
        winner_name = "SARIMA"
        winner_preds = p_sarima
        winner_r2 = sarima_metrics["R2"]
        
    print(f"\n>>> OVERALL WINNER: {winner_name} (R2={winner_r2:.4f})")
    
    plt.figure(figsize=(15, 6))
    # Plot tail of training for context
    plt.plot(train.index[-200:], train.iloc[-200:], label="History", color="gray", alpha=0.5)
    plt.plot(test.index, test, label="Actual Truth", color="black", linewidth=1.5)
    plt.plot(test.index, winner_preds, label=f"{winner_name} Forecast", color="red", linestyle="--", linewidth=2)
    
    plt.title(f"Best Model Forecast: {winner_name} (R2={winner_r2:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("best_model_forecast.png", dpi=300)
    print("Saved: best_model_forecast.png")

if __name__ == "__main__":
    train, test = load_data()
    
    best_arima = run_arima_study(train, test)
    best_sarima = run_sarima_study(train, test)
    
    compare_and_forecast(train, test, best_arima, best_sarima)