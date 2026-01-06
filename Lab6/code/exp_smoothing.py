from pandas import read_csv, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
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
SEASONAL_PERIODS = 96  # 24h * 4 (15-min intervals)

# --- Load pre-processed data (already scaled from Lab5) ---
train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime", parse_dates=True)
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime", parse_dates=True)

train: Series = train["Total"]
test: Series = test["Total"]

# --- ES Study ---
def es_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    model_types = [
        ("Simple ES", lambda data: SimpleExpSmoothing(data)),
        ("Holt's Linear", lambda data: ExponentialSmoothing(data, trend='add')),
        ("Holt-Winters", lambda data: ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=SEASONAL_PERIODS))
    ]
    
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ES", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(model_types), figsize=(len(model_types) * HEIGHT, HEIGHT))
    
    for i, (model_name, model_func) in enumerate(model_types):
        values = {}
        yvalues = []
        
        for alpha in alpha_values:
            try:
                model_instance = model_func(train)
                fitted_model = model_instance.fit(smoothing_level=alpha, optimized=True)
                prd_tst = fitted_model.forecast(len(test))
                
                eval_score: float = FORECAST_MEASURES[measure](test, prd_tst)
                print(f"{model_name} alpha={alpha} -> {measure}={eval_score:.4f}")
                
                if eval_score > best_performance and abs(eval_score - best_performance) > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["params"] = (model_name, alpha)
                    best_model = fitted_model
                yvalues.append(eval_score)
            except Exception as e:
                print(f"{model_name} alpha={alpha} -> FAILED: {e}")
                yvalues.append(None)
        
        values["alpha"] = yvalues
        plot_multiline_chart(
            alpha_values,
            values,
            ax=axs[i],
            title=f"{model_name} ({measure})",
            xlabel="alpha",
            ylabel=measure,
            percentage=flag,
        )
    
    print(
        f"ES best results achieved with model={best_params['params'][0]}, alpha={best_params['params'][1]} ==> {measure}={best_performance:.4f}"
    )
    
    return best_model, best_params

# --- Run Study ---
os.makedirs("Lab6/results/ES", exist_ok=True)

best_model, best_params = es_study(train, test, measure=measure)
savefig("Lab6/results/ES/parameters_study.png")

# --- Final Predictions ---
params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

# --- Plots ---
plot_forecasting_eval(
    train, test, prd_trn, prd_tst,
    title=f"{params[0]} (alpha={params[1]})"
)
savefig("Lab6/results/ES/evals.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{params[0]} (alpha={params[1]})",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
savefig("Lab6/results/ES/forecast.png")

print(f"\nSaved plots to Lab6/results/ES/")