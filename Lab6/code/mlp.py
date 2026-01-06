from pandas import read_csv, Series
from sklearn.neural_network import MLPRegressor
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
modules_path = os.path.join(project_root, 'package_lab')
if modules_path not in sys.path:
    sys.path.append(modules_path)
from dslabs_functions import HEIGHT, FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart
from matplotlib.pyplot import figure, savefig, subplots
from utils import plot_forecasting_eval, plot_forecasting_series

measure: str = "R2"

# --- Load pre-processed data (already scaled from Lab5) ---
train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime", parse_dates=True)
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime", parse_dates=True)

train: Series = train["Total"]
test: Series = test["Total"]

# --- Helper: Create lagged features ---
def create_lagged_features(series, n_lags):
    """Create X (lagged values) and y (target) for MLP"""
    X, y = [], []
    values = series.values
    for i in range(len(values) - n_lags):
        X.append(values[i:i + n_lags])
        y.append(values[i + n_lags])
    return np.array(X), np.array(y)

# --- MLP Study ---
def mlp_study(train: Series, test: Series, nr_episodes: int = 500, measure: str = "R2"):
    lag_values = [4, 12, 24, 48, 96]
    hidden_layer_configs = [(50,), (100,), (50, 25)]
    
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "MLP", "metric": measure, "params": ()}
    best_performance: float = -100000
    best_lag = 0

    fig, axs = subplots(1, len(lag_values), figsize=(len(lag_values) * HEIGHT, HEIGHT))
    
    for i, n_lags in enumerate(lag_values):
        X_train, y_train = create_lagged_features(train, n_lags)
        X_test, y_test = create_lagged_features(test, n_lags)
        
        values = {}
        for hidden in hidden_layer_configs:
            yvalues = []
            
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden,
                max_iter=nr_episodes,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            mlp.fit(X_train, y_train)
            prd_tst = mlp.predict(X_test)
            
            eval_score: float = FORECAST_MEASURES[measure](y_test, prd_tst)
            print(f"lag={n_lags} hidden={hidden} -> {measure}={eval_score:.4f}")
            
            if eval_score > best_performance and abs(eval_score - best_performance) > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (n_lags, hidden)
                best_model = mlp
                best_lag = n_lags
            yvalues.append(eval_score)
            values[str(hidden)] = yvalues
            
        # For plotting, we need consistent x-axis - use hidden config index
        plot_multiline_chart(
            [str(h) for h in hidden_layer_configs],
            {str(hidden_layer_configs[0]): [values[str(h)][0] for h in hidden_layer_configs]},
            ax=axs[i],
            title=f"MLP lag={n_lags} ({measure})",
            xlabel="hidden layers",
            ylabel=measure,
            percentage=flag,
        )
    
    print(
        f"MLP best results achieved with lag={best_params['params'][0]}, hidden={best_params['params'][1]} ==> {measure}={best_performance:.4f}"
    )
    
    return best_model, best_params, best_lag

# --- Run Study ---
os.makedirs("Lab6/results/MLP", exist_ok=True)

best_model, best_params, best_lag = mlp_study(train, test, nr_episodes=500, measure=measure)
savefig("Lab6/results/MLP/parameters_study.png")

# --- Final Predictions ---
params = best_params["params"]
X_train, y_train = create_lagged_features(train, best_lag)
X_test, y_test = create_lagged_features(test, best_lag)

prd_trn = best_model.predict(X_train)
prd_tst = best_model.predict(X_test)

# Convert to Series for plotting
train_trimmed = train.iloc[best_lag:]
test_trimmed = test.iloc[best_lag:]
prd_trn_series = Series(prd_trn, index=train_trimmed.index)
prd_tst_series = Series(prd_tst, index=test_trimmed.index)

# --- Plots ---
plot_forecasting_eval(
    train_trimmed, test_trimmed, prd_trn_series, prd_tst_series,
    title=f"MLP (lag={params[0]}, hidden={params[1]})"
)
savefig("Lab6/results/MLP/evals.png")

plot_forecasting_series(
    train_trimmed,
    test_trimmed,
    prd_tst_series,
    title=f"MLP (lag={params[0]}, hidden={params[1]})",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
savefig("Lab6/results/MLP/forecast.png")

print(f"\nSaved plots to Lab6/results/MLP/")