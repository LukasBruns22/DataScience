import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Updated paths for prepared data
TRAIN_FILENAME = "datasets/traffic_forecasting/processed_train.csv"
TEST_FILENAME = "datasets/traffic_forecasting/processed_test.csv"
TARGET = "Total"
SEASONAL_PERIODS = 96  # 24h * 4 (15min chunks) = 96 periods per day

def load_data():
    """Loads and preprocesses the prepared traffic data."""
    # Load Train
    print(f"Loading training data from {TRAIN_FILENAME}...")
    train_df = pd.read_csv(TRAIN_FILENAME, parse_dates=['Datetime'], index_col='Datetime')
    
    # Load Test
    print(f"Loading testing data from {TEST_FILENAME}...")
    test_df = pd.read_csv(TEST_FILENAME, parse_dates=['Datetime'], index_col='Datetime')
    
    # Ensure Frequency is set (Important for Statsmodels)
    if train_df.index.freq is None:
        train_df.index.freq = pd.infer_freq(train_df.index)
    if test_df.index.freq is None:
        test_df.index.freq = pd.infer_freq(test_df.index)
        
    return train_df[TARGET], test_df[TARGET]

def get_metrics(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def run_full_alpha_study():
    # 1. Load Data (Modified to load pre-split data)
    train, test = load_data()
    
    # (Removed manual 80/20 split since files are already split)
    
    print(f"Data Loaded. Train: {len(train)}, Test: {len(test)}")

    # 3. Define Model Constructors
    # We define functions that return a fresh unfitted model
    model_types = [
        ("Simple ES", lambda data: SimpleExpSmoothing(data)),
        ("Holt's Linear", lambda data: ExponentialSmoothing(data, trend='add')),
        ("Holt-Winters", lambda data: ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=SEASONAL_PERIODS))
    ]
    
    alpha_values = [i / 10.0 for i in range(1, 10)]
    
    results = []
    best_config = {"R2": -np.inf, "Model": None, "Pred": None, "Name": ""}

    print("\n--- Starting Alpha Study for All Models ---")
    
    # Loop over Models
    for name, model_func in model_types:
        print(f"Testing {name}...", end=" ")
        
        # Loop over Alphas
        for alpha in alpha_values:
            try:
                # Instantiate
                model_instance = model_func(train)
                
                # FIT: Fix Alpha, Optimize others (Beta/Gamma)
                # optimized=True allows statsmodels to find best Beta/Gamma given our fixed Alpha
                fitted_model = model_instance.fit(smoothing_level=alpha, optimized=True)
                
                # Forecast
                pred = fitted_model.forecast(len(test))
                
                # Evaluate
                metrics = get_metrics(test, pred, f"{name} (a={alpha})")
                
                # Store for plotting
                results.append({
                    "Model Type": name,
                    "Alpha": alpha,
                    "R2": metrics["R2"],
                    "MSE": metrics["MSE"]
                })
                
                # Check for absolute winner
                if metrics["R2"] > best_config["R2"]:
                    best_config["R2"] = metrics["R2"]
                    best_config["Model"] = fitted_model
                    best_config["Pred"] = pred
                    best_config["Name"] = f"{name} (alpha={alpha})"
                    
            except Exception as e:
                pass # Skip convergence failures
        print("Done.")

    results_df = pd.DataFrame(results)

    # --- PLOT 1: ALPHA SENSITIVITY STUDY ---
    print("\nGenerating Alpha Study Plot...")
    plt.figure(figsize=(12, 6))
    
    # We plot one line per model type
    sns.lineplot(data=results_df, x="Alpha", y="R2", hue="Model Type", marker="o", linewidth=2)
    
    plt.title("Hyperparameter Study: Effect of Alpha on Model Performance (R2)")
    plt.xlabel("Alpha (Smoothing Level)")
    plt.ylabel("R2 Score (Higher is Better)")
    plt.grid(True, alpha=0.3)
    plt.savefig('ES_Alpha_Study_All_Models.png')
    print("Saved: ES_Alpha_Study_All_Models.png")

    # --- PLOT 2: BEST MODEL FORECAST ---
    print(f"Generating Forecast Plot for Winner: {best_config['Name']}...")
    
    plt.figure(figsize=(15, 6))
    # Last 5 days of training for context (Using 96 periods/day * 5 days = 480)
    plot_train = train.iloc[-480:]
    
    plt.plot(plot_train.index, plot_train, label='Training Data (Last 5 Days)')
    plt.plot(test.index, test, label='Actual Test Data', color='black', alpha=0.5)
    plt.plot(test.index, best_config["Pred"], label=f'Forecast', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Best Model Forecast: {best_config['Name']} (R2: {best_config['R2']:.3f})")
    plt.xlabel("Date")
    plt.ylabel(TARGET)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ES_Best_Forecast.png')
    print("Saved: ES_Best_Forecast.png")
    
    # --- FINAL SUMMARY ---
    print("\n--- Top 5 Configurations ---")
    print(results_df.sort_values(by="R2", ascending=False).head(5))

if __name__ == "__main__":
    run_full_alpha_study()