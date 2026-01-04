from sklearn.base import RegressorMixin
from pandas import read_csv, DataFrame, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from utils import plot_forecasting_eval, plot_forecasting_series
from matplotlib.pyplot import figure, savefig
from numpy import mean
import pandas as pd

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
modules_path = os.path.join(project_root, 'package_lab')
if modules_path not in sys.path:
    sys.path.append(modules_path)
from dslabs_functions import HEIGHT, FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart

measure: str = "R2"

def exponential_smoothing_study(train: Series, test: Series, measure: str = "R2"):
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (alpha,)
            best_model = model
        yvalues.append(eval)

    print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing ({measure})",
        xlabel="alpha",
        ylabel=measure,
        percentage=flag,
    )

    return best_model, best_params

train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime", parse_dates=True)
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime", parse_dates=True)

train: Series = train["Total"]
test: Series = test["Total"]

best_model, best_params = exponential_smoothing_study(train, test, measure=measure)
savefig(f"Lab6/results/ES/parameters_study.png")

params = best_params["params"]
prd_trn = best_model.predict(start=0, end=len(train) - 1)
prd_tst = best_model.forecast(steps=len(test))

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Exponential Smoothing alpha={params[0]}")
savefig(f"Lab6/results/ES/evals.png")


plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Exponential Smoothing (alpha={params[0]})",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
savefig(f"Lab6/results/ES/forecast.png")