from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show, subplot, suptitle, tight_layout
from utils.dslabs_functions import (
    CLASS_EVAL_METRICS, DELTA_IMPROVE, 
    plot_bar_chart, plot_evaluation_results, 
    read_train_test_from_files
)

TRAIN_FILENAME = "datasets/traffic_accidents_prepared_train.csv"
TEST_FILENAME = "datasets/traffic_accidents_prepared_test.csv"

TARGET = "crash_type_enc"
EVAL_METRIC = "f1"
FILE_TAG = "traffic_accidents"


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

def plot_NB_results(model_names, scores, save_path="graphs/NB/{FILE_TAG}_nb_metrics_study"):
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

def plot_best_NB_model(best_model, params, labels, trnX, tstX, trnY, tstY, save_path):
    if save_path is None:
        save_path = f"graphs/NB/traffic_accidents_{params['name']}_best_{params['metric']}_eval"
    
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'{save_path}.png')


def main():
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        TRAIN_FILENAME, TEST_FILENAME, TARGET
    )

    model_names, scores, best_model, params = naive_Bayes_study(
        trnX, trnY, tstX, tstY, metric_for_best=EVAL_METRIC
    )

    plot_NB_results(model_names, scores)
    plot_best_NB_model(best_model, params, labels, trnX, tstX, trnY, tstY, "graphs/NB/traffic_accidents_NB_best_model")


if __name__ == "__main__":
    main()