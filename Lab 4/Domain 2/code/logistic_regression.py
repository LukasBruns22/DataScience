from numpy import array, ndarray, where, random, concatenate, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.linear_model import LogisticRegression
from utils.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from utils.dslabs_functions import plot_evaluation_results, plot_multiline_chart

TRAIN_FILENAME = "datasets/combined_flights_prepared_train.csv"
TEST_FILENAME = "datasets/combined_flights_prepared_test.csv"

TARGET = "Cancelled"
EVAL_METRIC = "f1"
FILE_TAG = "combined_flights"

def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[LogisticRegression | None, dict]:

    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    penalty_types: list[str] = ["l1", "l2"]

    best_model = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for type in penalty_types:
        warm_start = False
        y_tst_values: list[float] = []
        for j in range(len(nr_iterations)):
            clf = LogisticRegression(
                penalty=type,
                max_iter=lag,
                warm_start=warm_start,
                solver="liblinear",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            warm_start = True
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params["params"] = (type, nr_iterations[j])
                best_model: LogisticRegression = clf
        values[type] = y_tst_values
    
    figure()
    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"graphs/LR/{FILE_TAG}_lr_{EVAL_METRIC}_study.png")
    return best_model, best_params

def overfitting_study(params: dict):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        TRAIN_FILENAME, TEST_FILENAME, TARGET
    )

    type: str = params["params"][0]
    nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    
    warm_start = False
    for n in nr_iterations:
        clf = LogisticRegression(
            warm_start=warm_start,
            penalty=type,
            max_iter=n,
            solver="liblinear",
            verbose=False,
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](trnY, prd_trn_Y))
        warm_start = True

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"LR overfitting study for penalty={type}",
        xlabel="nr_iterations",
        ylabel=str(EVAL_METRIC),
        percentage=True,
    )
    savefig(f"graphs/LR/{FILE_TAG}_lr_{EVAL_METRIC}_overfitting.png")


def main():
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

    best_model, params = logistic_regression_study(
        trnX, trnY, tstX, tstY,
        nr_max_iterations=1000, lag=100, metric=EVAL_METRIC
    )
    
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'graphs/LR/{FILE_TAG}_lr_{params["name"]}_best_{params["metric"]}_eval.png')
    
    overfitting_study(params)

if __name__ == "__main__":
    main()


