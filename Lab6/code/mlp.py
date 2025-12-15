import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 0. Setup
# ==========================================
PLOT_DIR = "plots/plots_mlp"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================
# 1. Load Data
# ==========================================
# Assuming files exist at these paths
train_df = pd.read_csv(
    'datasets/traffic_forecasting/processed_train.csv',
    parse_dates=['Datetime'], index_col='Datetime'
)

test_df = pd.read_csv(
    'datasets/traffic_forecasting/processed_test.csv',
    parse_dates=['Datetime'], index_col='Datetime'
)

# ==========================================
# 2. Scaling (Fit on Train, Apply to Test)
# ==========================================
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[['Total']])
test_scaled = scaler.transform(test_df[['Total']])

# ==========================================
# 3. Helper: Windowing Function
# ==========================================
def create_lagged_features(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps].flatten())
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# ==========================================
# 4. Define Hyperparameter Space
# ==========================================
# 
window_sizes = [12, 24, 48, 96, 192, 384, 768, 1536]

hidden_layers_configs = [(64,), (128,), (64, 32)] 
learning_rates = [0.001, 0.01]

results_list = []
best_score = float('inf')
best_model = None
best_params = {}
best_window_size = 0

print(f"Starting Study...")
print(f"Configs: {len(window_sizes)} Windows x {len(hidden_layers_configs)} Layers x {len(learning_rates)} LRs")
print("-" * 50)

# ==========================================
# 5. Custom Hyperparameter Loop
# ==========================================

for w_size in window_sizes:
    X_train, y_train = create_lagged_features(train_scaled, w_size)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for layers in hidden_layers_configs:
        for lr in learning_rates:
            
            # B. Define Model
            # 
            mlp = MLPRegressor(
                hidden_layer_sizes=layers,
                learning_rate_init=lr,
                activation='relu',
                solver='adam',     
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
            # C. Train & Evaluate using Cross Validation
            cv_scores = []
            
            for train_index, val_index in tscv.split(X_train):
                X_t, X_v = X_train[train_index], X_train[val_index]
                y_t, y_v = y_train[train_index], y_train[val_index]
                
                mlp.fit(X_t, y_t.ravel())
                pred_v = mlp.predict(X_v)
                mse_val = mean_squared_error(y_v, pred_v)
                cv_scores.append(mse_val)
            
            avg_mse = np.mean(cv_scores)
            
            # Store results
            model_name = f"W={w_size} | L={layers} | LR={lr}"
            print(f"Checked: {model_name} -> CV MSE: {avg_mse:.4f}")
            
            results_list.append({
                "Window_Size": w_size,
                "Hidden_Layers": str(layers),
                "Learning_Rate": lr,
                "CV_MSE": avg_mse,
                "Model_Label": model_name
            })
            
            # Check if this is the new best model
            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {
                    "window_size": w_size,
                    "hidden_layer_sizes": layers,
                    "learning_rate_init": lr
                }
                # Save the input dimension needed for the final test
                best_window_size = w_size

# ==========================================
# 6. Process Results Table
# ==========================================
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(by="CV_MSE", ascending=True)

print("\nTOP 5 MODELS:")
print(results_df.head(5))

# Save table
results_df.to_csv("hyperparameter_study_results.csv", index=False)

# ==========================================
# 7. Plot 1: Hyperparameter Performance Comparison
# ==========================================
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Model_Label", y="CV_MSE", palette="viridis")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.title(f"Model Comparison (Window Size vs Layers vs LR)")
plt.ylabel("Mean Squared Error (CV)")
plt.xlabel("Configuration")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/model_comparison_bar.png", dpi=200)
plt.close()
print(f"Saved: {PLOT_DIR}/model_comparison_bar.png")

# ==========================================
# 8. Retrain Best Model on Full Train Data
# ==========================================
print("\nRetraining Best Model on full training set...")
print(f"Best Config: {best_params}")

# Re-create data with the WINNING window size
X_train_best, y_train_best = create_lagged_features(train_scaled, best_window_size)
X_test_best, y_test_best = create_lagged_features(test_scaled, best_window_size)

final_model = MLPRegressor(
    hidden_layer_sizes=best_params["hidden_layer_sizes"],
    learning_rate_init=best_params["learning_rate_init"],
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

final_model.fit(X_train_best, y_train_best.ravel())

# ==========================================
# 9. Final Evaluation
# ==========================================
# Predict
y_pred_scaled = final_model.predict(X_test_best)

# Inverse Transform
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_best.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)

print("\n========== BEST MODEL FINAL METRICS ==========")
print(f"Window Size: {best_window_size}")
print(f"Structure:   {best_params['hidden_layer_sizes']}")
print(f"LR:          {best_params['learning_rate_init']}")
print("-" * 30)
print(f"R2 Score: {r2:.4f}")
print(f"MAE:      {mae:.4f}")
print(f"MSE:      {mse:.4f}")
print(f"RMSE:     {rmse:.4f}")
print("==============================================\n")

# ==========================================
# 10. Plot 2: Best Model Forecast
# ==========================================
# Align dates (trim the first N_STEPS from test index)
test_dates = test_df.index[best_window_size:]

plt.figure(figsize=(15, 6))
plt.plot(test_dates, y_test_actual, label="Actual Traffic", color='black', alpha=0.7, linewidth=1.5)
plt.plot(test_dates, y_pred, label="MLP Prediction", color='#2ca02c', linestyle='--', linewidth=2)

plt.title(f"Best Model Forecast (Window={best_window_size}, R2={r2:.2f})")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{PLOT_DIR}/best_model_forecast.png", dpi=200)
plt.close()
print(f"Saved: {PLOT_DIR}/best_model_forecast.png")