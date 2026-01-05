import json
import os
import numpy as np
from numpy import array, ndarray, where, random, concatenate, unique
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, subplot, suptitle, tight_layout, subplots
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.dslabs_functions import (
    CLASS_EVAL_METRICS, DELTA_IMPROVE, HEIGHT,
    plot_bar_chart, plot_evaluation_results, plot_multiline_chart,
    read_train_test_from_files
)

TRAIN_FILENAME = "datasets/combined_flights_prepared_train.csv"
TEST_FILENAME = "datasets/combined_flights_prepared_test.csv"
TARGET = "Cancelled"
EVAL_METRIC = "f1"
FILE_TAG = "combined_flights"

# ---------------------------------------------------------
# Basic Naive Bayes Study
# ---------------------------------------------------------

def naive_Bayes_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    metric_for_best: str = "accuracy",
):
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }

    metrics = ["precision", "accuracy", "recall"]

    model_names: list[str] = list(estimators.keys())
    scores: dict[str, list[float]] = {m: [] for m in metrics}

    best_model = None
    best_params: dict = {"name": "", "metric": metric_for_best, "params": ()}
    best_performance: float = 0.0

    for name, clf in estimators.items():
        print(f"NB {name}")
        clf.fit(trnX, trnY)
        prdY: array = clf.predict(tstX)

        # store the three metrics for plotting
        for m in metrics:
            value: float = CLASS_EVAL_METRICS[m](tstY, prdY)
            scores[m].append(value)

        # use metric_for_best (e.g. "f1") to choose best model
        best_value = CLASS_EVAL_METRICS[metric_for_best](tstY, prdY)
        if best_value - best_performance > DELTA_IMPROVE:
            best_performance = best_value
            best_params["name"] = name
            best_params[metric_for_best] = best_value
            best_model = clf

    return model_names, scores, best_model, best_params

def plot_NB_results(model_names, scores, save_path=f"graphs/NB/{FILE_TAG}_nb_metrics_study"):
    figure(figsize=(12, 4))

    # Subplot 1: precision
    subplot(1, 3, 1)
    plot_bar_chart(
        model_names,
        scores["precision"],
        title="Naive Bayes (precision)",
        ylabel="precision",
        percentage=True,
    )

    # Subplot 2: accuracy
    subplot(1, 3, 2)
    plot_bar_chart(
        model_names,
        scores["accuracy"],
        title="Naive Bayes (accuracy)",
        ylabel="accuracy",
        percentage=True,
    )

    # Subplot 3: recall
    subplot(1, 3, 3)
    plot_bar_chart(
        model_names,
        scores["recall"],
        title="Naive Bayes (recall)",
        ylabel="recall",
        percentage=True,
    )

    suptitle("Naive Bayes Models – Precision, Accuracy, Recall")
    tight_layout(rect=[0, 0, 1, 0.90])

    savefig(f"{save_path}.png")

def plot_best_NB_model(best_model, params, labels, trnX, tstX, trnY, tstY):
    save_path = f"graphs/NB/{FILE_TAG}_{params['name']}_best_{params['metric']}_eval"
    
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'{save_path}.png')


# ---------------------------------------------------------
# Hyperparameter Study
# ---------------------------------------------------------

def evaluate_nb_model(
    model_class,
    param_name: str,
    param_values: list,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print(f"\nEvaluating model {model_class.__name__} ({param_name})")

    accuracies, precisions, recalls = [], [], []

    for val in param_values:
        print(f"Testing {param_name} = {val:.2e}")

        model = model_class(**{param_name: val})
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

        print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    print(f"Finished {model_class.__name__}\n")

    return {
        "Accuracy": accuracies,
        "Precision": precisions,
        "Recall": recalls,
    }

def get_best_config(model_name: str, param_values: list, results: dict) -> dict:
    accuracies = results["Accuracy"]
    best_idx = int(np.argmax(accuracies))

    return {
        "model": model_name,
        "hyperparameter": float(param_values[best_idx]),
        "accuracy": float(results["Accuracy"][best_idx]),
        "precision": float(results["Precision"][best_idx]),
        "recall": float(results["Recall"][best_idx]),
    }

def run_hyperparameter_study(trnX, trnY, tstX, tstY):
    print("Starting Naive Bayes hyperparameter study")

    print("\nFeature matrix shapes:")
    print(f"Train: {trnX.shape}")
    print(f"Test : {tstX.shape}")

    fig, axs = subplots(1, 2, figsize=(15, HEIGHT), squeeze=False)
    
    # Hyperparameter ranges
    gaussian_vs = np.logspace(-15, -3, 20)
    bernoulli_alpha = np.logspace(-15, -3, 20)

    ax_gaussian = axs[0, 0]
    ax_bernoulli = axs[0, 1]

    # GaussianNB
    gaussian_results = evaluate_nb_model(
        GaussianNB,
        "var_smoothing",
        gaussian_vs.tolist(),
        trnX,
        trnY,
        tstX,
        tstY,
    )

    # BernoulliNB
    bernoulli_results = evaluate_nb_model(
        BernoulliNB,
        "alpha",
        bernoulli_alpha.tolist(),
        trnX,
        trnY,
        tstX,
        tstY,
    )

    # -------------------------------------------------
    # Save best configurations to JSON
    # -------------------------------------------------
    best_models = [
        get_best_config("GaussianNB", gaussian_vs.tolist(), gaussian_results),
        get_best_config("BernoulliNB", bernoulli_alpha.tolist(), bernoulli_results),
    ]
    
    # Ensure directory exists
    os.makedirs("results/NB", exist_ok=True)
    
    json_path = "results/NB/best_nb_models.json"
    with open(json_path, "w") as f:
        json.dump(best_models, f, indent=4)

    print(f"Best model configurations saved to: {json_path}")

    # Gaussian subplot
    plot_multiline_chart(
        xvalues=gaussian_vs.tolist(),
        yvalues=gaussian_results,
        ax=ax_gaussian,
        title="GaussianNB Hyperparameter Study",
        xlabel="var_smoothing",
        ylabel="Metric value",
        percentage=False,
    )
    ax_gaussian.set_xscale("log")

    # Bernoulli subplot
    plot_multiline_chart(
        xvalues=bernoulli_alpha.tolist(),
        yvalues=bernoulli_results,
        ax=ax_bernoulli,
        title="BernoulliNB Hyperparameter Study",
        xlabel="alpha",
        ylabel="Metric value",
        percentage=False,
    )
    # ax_bernoulli.set_xscale("log") # alpha for BernoulliNB expects floats, usually small but can be linear. Log scale is fine.

    fig.tight_layout()
    savefig(f"graphs/NB/{FILE_TAG}_hyperparameter_study.png")

    print("All studies completed successfully!")
    print(f"Figure saved to: graphs/NB/{FILE_TAG}_hyperparameter_study.png")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(undersample: bool = False):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        TRAIN_FILENAME, TEST_FILENAME, TARGET
    )

    if undersample:
        FILE_TAG = "combined_flights_undersampled"
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

    # 1. Basic Study
    print("Running Basic Naive Bayes Study...")
    model_names, scores, best_model, params = naive_Bayes_study(
        trnX, trnY, tstX, tstY, metric_for_best=EVAL_METRIC
    )

    plot_NB_results(model_names, scores)
    plot_best_NB_model(best_model, params, labels, trnX, tstX, trnY, tstY)
    
    # 2. Hyperparameter Study
    print("\nRunning Hyperparameter Study...")
    run_hyperparameter_study(trnX, trnY, tstX, tstY)

if __name__ == "__main__":
    main(undersample=False)