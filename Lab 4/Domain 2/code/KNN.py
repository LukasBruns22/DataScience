from typing import Literal
from numpy import array, ndarray, where, random, concatenate, unique
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from sklearn.model_selection import train_test_split
import time
from utils.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from utils.dslabs_functions import read_train_test_from_files, plot_evaluation_results

TRAIN_FILENAME = "datasets/combined_flights_prepared_train.csv"
TEST_FILENAME = "datasets/combined_flights_prepared_test.csv"

TARGET = "Cancelled"
EVAL_METRIC = "f1"
FILE_TAG = "combined_flights"

K_MAX = 31
LAG = 2

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='f1'
        ) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

    kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params['params'] = (k, d)
                best_model = clf
            print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params

def overfitting_study(params: dict, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array):
    distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
    kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
    y_tst_values: list = []
    y_trn_values: list = []

    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        kvalues,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"KNN overfitting study for {distance}",
        xlabel="K",
        ylabel=str(EVAL_METRIC),
        percentage=True,
    )
    savefig(f"graphs/KNN/{FILE_TAG}_knn_overfitting.png")
    print(f"KNN overfitting study saved for {distance}")
    return

def run_knn_study(k_max: int=25, lag: int=2):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(TRAIN_FILENAME, TEST_FILENAME, TARGET)
    print(f'Train#={len(trnX)} Test#={len(tstX)}')
    print(f'Labels={labels}')

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


    figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max, lag, metric=EVAL_METRIC)
    savefig(f'graphs/KNN/{FILE_TAG}_knn_{EVAL_METRIC}_study.png')

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'graphs/KNN/{FILE_TAG}_knn_{params["name"]}_best_{params["metric"]}_eval.png')

    overfitting_study(params, trnX, trnY, tstX, tstY)
    print(f"KNN best model: {best_model}")
    print(f"KNN best params: {params}")
    


def main():
    run_knn_study(K_MAX, LAG)


if __name__ == "__main__":
    main()