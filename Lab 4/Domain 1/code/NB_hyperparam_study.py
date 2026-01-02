import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from matplotlib.pyplot import figure, subplots, show
from numpy import array, ndarray
from pandas import DataFrame, read_csv
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.dslabs_functions import plot_multiline_chart, HEIGHT, read_train_test_from_files
import json
import os

TRAIN_FILENAME = "datasets/traffic_accidents_prepared_train.csv"
TEST_FILENAME = "datasets/traffic_accidents_prepared_test.csv"
TARGET = "crash_type_enc"
EVAL_METRIC = "f1"
FILE_TAG = "traffic_accidents"


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

# ---------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------
def run_all_nb_studies():
    print("Starting Naive Bayes hyperparameter study")

    X_train, X_test, y_train, y_test, _, _ = read_train_test_from_files(
        TRAIN_FILENAME, TEST_FILENAME, TARGET
    )

    print("\nFeature matrix shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}")

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
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # BernoulliNB
    bernoulli_results = evaluate_nb_model(
        BernoulliNB,
        "alpha",
        bernoulli_alpha.tolist(),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # -------------------------------------------------
    # Save best configurations to JSON
    # -------------------------------------------------
    best_models = [
        get_best_config("GaussianNB", gaussian_vs.tolist(), gaussian_results),
        get_best_config("BernoulliNB", bernoulli_alpha.tolist(), bernoulli_results),
    ]
    
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
    ax_bernoulli.set_xscale("log")

    fig.tight_layout()
    fig.savefig("graphs/NB/hyperparameter_study.png")

    print("All studies completed successfully!")
    print("Figure saved to: graphs/NB/hyperparameter_study.png")


# ---------------------------------------------------------
# Run program
# ---------------------------------------------------------
if __name__ == "__main__":
    run_all_nb_studies()