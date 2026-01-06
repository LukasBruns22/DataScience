import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin
from pandas import Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
modules_path = os.path.join(project_root, 'package_lab')
if modules_path not in sys.path:
    sys.path.append(modules_path)
from dslabs_functions import plot_forecasting_eval, plot_forecasting_series

FORECAST_MEASURES = {
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "MAPE": mean_absolute_percentage_error,
}

class PersistenceRealistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0
        self.estimations = [0]
        self.obs_len = 0

    def fit(self, X: Series):
        for i in range(1, len(X)):
            self.estimations.append(X.iloc[i - 1])
        self.obs_len = len(self.estimations)
        self.last = X.iloc[len(X) - 1]
        prd_series: Series = Series(self.estimations)
        prd_series.index = X.index
        return prd_series

    def predict(self, X: Series):
        prd: list = len(X) * [self.last]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series

def train_and_evaluate(train, test, model_type, approach_name, target_col='Total'):
    """
    Trains a model and evaluates it.
    For multivariate data, uses the first column (Total) for evaluation.
    """
    
    # Handle multivariate: extract target column for univariate models
    if isinstance(train, pd.DataFrame) and len(train.columns) > 1:
        train_series = train[target_col]
        test_series = test[target_col]
    elif isinstance(train, pd.DataFrame):
        train_series = train.iloc[:, 0]
        test_series = test.iloc[:, 0]
    else:
        train_series = train
        test_series = test
    
    print(f"\n=== Training {model_type.upper()} | {approach_name} ===")

    # Fit Model and Predict
    if model_type == 'Persistence':
        model = PersistenceRealistRegressor()
        model.fit(train_series)
        prd_trn = model.predict(train_series)
        prd_tst = model.predict(test_series)
    elif model_type == 'LR':
        model = LinearRegression()
        trnX = np.arange(len(train_series)).reshape(-1, 1)
        trnY = train_series.to_numpy()
        tstX = np.arange(len(train_series), len(train_series)+len(test_series)).reshape(-1, 1)
        model.fit(trnX, trnY)
        prd_trn = Series(model.predict(trnX).flatten(), index=train_series.index)
        prd_tst = Series(model.predict(tstX).flatten(), index=test_series.index)
    else:
        raise ValueError("model_type must be 'Persistence' or 'LR'")

    # Generate Plots
    plot_forecasting_eval(train_series, test_series, prd_trn, prd_tst, title=f"{approach_name} - {model_type}")
    plt.savefig(f"Lab5/results/{model_type}_{approach_name}_metrics.png")

    plot_forecasting_series(
        train_series,
        test_series,
        prd_tst,
        title=f"{approach_name} - {model_type}",
        xlabel='timestamp',
        ylabel='Total',
    )
    plt.savefig(f"Lab5/results/{model_type}_{approach_name}_forecast.png")
    
    # Calculate Metrics
    results = {'approach': approach_name, 'model_type': model_type}
    for measure_name, measure_func in FORECAST_MEASURES.items():
        trn_score = measure_func(train_series, prd_trn)
        tst_score = measure_func(test_series, prd_tst)
        results[measure_name] = tst_score
        print(f"{measure_name} - Train: {trn_score:.4f} | Test: {tst_score:.4f}")

    # Return metrics for comparison
    return results

def compare_strategies_averaged(results_list, current_baseline_score, metric='MAE'):
    """
    Groups results by 'approach', calculates the MEAN score of models (Persistence+LR),
    and identifies the best data preprocessing strategy.

    Parameters:
    -----------
    results_list : list of dict
        Collected results from training.
    metric : str
        The metric to average ('f1_score' or 'accuracy').
    """
    print("\n" + "="*40)
    print(f"   FINAL COMPARISON (Averaged by Strategy)")
    print("="*40)
    
    # Convert list of dicts to DataFrame for easy grouping
    df_results = pd.DataFrame(results_list)
    
    if df_results.empty:
        print("No results to compare.")
        return None

    summary = df_results.groupby('approach')[['MAE', 'MSE', 'R2']].mean()
    
    summary = summary.sort_values(by=metric, ascending=True if metric != 'R2' else False)
    
    print(f"\nRanking based on Average {metric.upper()}:\n")
    print(summary)
    
    best_approach = summary.index[0]
    best_score = summary.iloc[0][metric]

    if best_score > current_baseline_score:
        print(f"\nNo strategy outperformed the baseline score of {current_baseline_score:.4f}. Retaining baseline.")
        return best_approach, current_baseline_score, True
    
    print(f"\n>>> BEST STRATEGY: '{best_approach}'")
    print(f"   -> Avg {metric}: {best_score:.4f}")
    
    return best_approach, best_score, False