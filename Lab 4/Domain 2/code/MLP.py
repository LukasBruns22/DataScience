from typing import Literal
from numpy import array, ndarray, arange, where, random, concatenate, unique

from pandas import DataFrame, concat
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.neural_network import MLPClassifier
from utils.dslabs_functions import (CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files)
from utils.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart, plot_line_chart

TRAIN_FILENAME = "datasets/combined_flights_prepared_train.csv"
TEST_FILENAME = "datasets/combined_flights_prepared_test.csv"
TARGET = "Cancelled"
EVAL_METRIC = "f1"
FILE_TAG = "combined_flights"

def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 5000,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    figure()
    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    if best_params["params"]:
        print(
            f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]})'
        )
    else:
        print("MLP: No model better than baseline found.")

    return best_model, best_params

def overfitting_study(params: dict, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, lag: int = 500, nr_max_iter: int = 2500):
    lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
    lr: float = params["params"][1]

    nr_iterations: list[int] = [lag] + [i for i in range(2 * lag, nr_max_iter + 1, lag)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric = "accuracy"

    warm_start: bool = False
    for n in nr_iterations:
        clf = MLPClassifier(
            warm_start=warm_start,
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=n,
            activation="logistic",
            solver="sgd",
            verbose=False,
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
        warm_start = True

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
        xlabel="nr_iterations",
        ylabel=str(EVAL_METRIC),
        percentage=True,
    )
    savefig(f"graphs/MLP/{FILE_TAG}_mlp_{EVAL_METRIC}_overfitting.png")

def loss_curve_study(best_model: MLPClassifier):
    figure()
    plot_line_chart(
        arange(len(best_model.loss_curve_)),
        best_model.loss_curve_,
        title="Loss curve for MLP best model training",
        xlabel="iterations",
        ylabel="loss",
        percentage=False,
    )
    savefig(f"graphs/MLP/{FILE_TAG}_mlp_{EVAL_METRIC}_loss_curve.png")

def run_mlp_study(nr_max_iterations: int = 5000, lag: int = 500):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        TRAIN_FILENAME, TEST_FILENAME, TARGET
    )
    
    # Sampling Logic
    trnY = array(trnY)
    n_neg = len(where(trnY == 0)[0])
    n_pos = len(where(trnY == 1)[0])
    print(f"Original distribution: 0={n_neg}, 1={n_pos}")
    
    # We undersample the majority class (0) to match the minority class (1)
    neg_indices = where(trnY == 0)[0]
    pos_indices = where(trnY == 1)[0]
    
    random.seed(42) # Ensure reproducibility
    sampled_neg_indices = random.choice(neg_indices, size=n_pos, replace=False)
    
    balanced_indices = concatenate([sampled_neg_indices, pos_indices])
    random.shuffle(balanced_indices)
    
    trnX = trnX[balanced_indices]
    trnY = trnY[balanced_indices]
    
    print(f"Sampled distribution: {unique(trnY, return_counts=True)}")
    # End Sampling Logic
    
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")
    
    best_model, params = mlp_study(trnX, trnY, tstX, tstY, nr_max_iterations, lag, metric=EVAL_METRIC)
    savefig(f"graphs/MLP/{FILE_TAG}_mlp_{EVAL_METRIC}_study.png")

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'graphs/MLP/{FILE_TAG}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')

    overfitting_study(params, trnX, trnY, tstX, tstY, lag, nr_max_iterations)
    print(f"MLP best model: {best_model}")
    print(f"MLP best params: {params}")

    loss_curve_study(best_model)

if __name__ == "__main__":
    run_mlp_study(nr_max_iterations=2000, lag=250)