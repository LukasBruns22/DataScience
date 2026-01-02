from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import RandomForestClassifier
from utils.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from utils.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart, plot_horizontal_bar_chart

TRAIN_FILENAME = "datasets/traffic_accidents_prepared_train.csv"
TEST_FILENAME = "datasets/traffic_accidents_prepared_test.csv"
TARGET = "crash_type_enc"
EVAL_METRIC = "f1"
FILE_TAG = "traffic_accidents"

def random_forests_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = RandomForestClassifier(
                    n_estimators=n, max_depth=d, max_features=f
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, f, n)
                    best_model = clf
                print(f'RF d={d} f={f} n={n}')
            values[f] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model, best_params

def feature_importance_study(vars: list[str], best_model: RandomForestClassifier):
    stdevs: list[float] = list(
        std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    )
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure(figsize=(8, 10))
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="RF variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"graphs/RandomForest/{FILE_TAG}_rf_{EVAL_METRIC}_vars_ranking.png")

def overfitting_study(params: dict, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array):
    d_max: int = params["params"][0]
    feat: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"RF overfitting study for d={d_max} and f={feat}",
        xlabel="nr_estimators",
        ylabel=str(EVAL_METRIC),
        percentage=True,
    )
    savefig(f"graphs/RandomForest/{FILE_TAG}_rf_{EVAL_METRIC}_overfitting.png")

def run_random_forest_study(nr_max_trees: int = 1000, lag: int = 250):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(TRAIN_FILENAME, TEST_FILENAME, TARGET)
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")

    best_model, params = random_forests_study(trnX, trnY, tstX, tstY, nr_max_trees, lag, metric=EVAL_METRIC)
    savefig(f"graphs/RandomForest/{FILE_TAG}_rf_{EVAL_METRIC}_study.png")

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'graphs/RandomForest/{FILE_TAG}_rf_{params["name"]}_best_{params["metric"]}_eval.png')

    feature_importance_study(vars, best_model)
    print(f"RF best model: {best_model}")
    print(f"RF best params: {params}")

    overfitting_study(params, trnX, trnY, tstX, tstY)

if __name__ == "__main__":
    run_random_forest_study(nr_max_trees=1000, lag=250)