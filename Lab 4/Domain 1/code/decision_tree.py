from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from utils.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
from utils.dslabs_functions import plot_evaluation_results, plot_multiline_chart, plot_horizontal_bar_chart
from sklearn.tree import export_graphviz
from matplotlib.pyplot import imread, imshow, axis
from subprocess import call
from sklearn.tree import plot_tree
from numpy import argsort

TRAIN_FILENAME = "datasets/traffic_accidents_prepared_train.csv"
TEST_FILENAME = "datasets/traffic_accidents_prepared_test.csv"
TARGET = "crash_type_enc"
EVAL_METRIC = "f1"
FILE_TAG = "traffic_accidents"

def trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int=10, lag:int=2, metric='f1'
        ) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max+1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            print(f'DT {c} and d={d}')
        values[c] = y_tst_values
        
    print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
    
    figure()
    plot_multiline_chart(
        depths, values, 
        title=f'DT Models ({metric})', 
        xlabel='d', 
        ylabel=metric, 
        percentage=True
    )

    return best_model, best_params

def variable_importance(labels: list[str], vars: list[str], best_model: DecisionTreeClassifier):
    tree_filename: str = f"graphs/DT/{FILE_TAG}_dt_{EVAL_METRIC}_best_tree"
    max_depth2show = 3
    st_labels: list[str] = [str(value) for value in labels]

    # Generate DOT file
    export_graphviz(
        best_model,
        out_file=tree_filename + ".dot",
        max_depth=max_depth2show,
        feature_names=vars,
        class_names=st_labels,
        filled=True,
        rounded=True,
        impurity=False,
        special_characters=True,
        precision=2,
    )

    # Generate high DPI Graphviz PNG
    call([
        "dot",
        "-Gdpi=600",
        "-Tpng",
        tree_filename + ".dot",
        "-o",
        tree_filename + "_gv.png"
    ])

    # Generate Matplotlib rendering (for interactive plots / reports)
    figure(figsize=(18, 10))
    plot_tree(
        best_model,
        max_depth=max_depth2show,
        feature_names=vars,
        class_names=st_labels,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
    )

    # Save as separate file with explicit DPI
    savefig(tree_filename + "_mpl.png", dpi=300)
    axis("off")


def feature_importance(vars: list[str], best_model: DecisionTreeClassifier):
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure(figsize=(10, 12))   # wider and much taller
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Decision Tree variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"graphs/DT/{FILE_TAG}_dt_{EVAL_METRIC}_vars_ranking.png")

def overfitting_study(params: dict, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int):
    crit: Literal["entropy", "gini"] = params["params"][0]
    depths: list[int] = [i for i in range(2, d_max + 1, 1)]
    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[EVAL_METRIC](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study for {crit}",
        xlabel="max_depth",
        ylabel=str(EVAL_METRIC),
        percentage=True,
    )
    savefig(f"graphs/DT/{FILE_TAG}_dt_{EVAL_METRIC}_overfitting.png")

def run_trees_study(d_max: int=25, lag:int=2):
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(TRAIN_FILENAME, TEST_FILENAME, TARGET)
    print(f'Train#={len(trnX)} Test#={len(tstX)}')
    print(f'Labels={labels}')

    best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max, lag, metric=EVAL_METRIC)
    savefig(f'graphs/DT/{FILE_TAG}_dt_{EVAL_METRIC}_study.png')

    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'graphs/DT/{FILE_TAG}_dt_{params["name"]}_best_{params["metric"]}_eval.png')

    variable_importance(labels, vars, best_model)
    print(f"DT best model: {best_model}")
    print(f"DT best params: {params}")

    feature_importance(vars, best_model)
    print(f"DT feature importance saved")

    overfitting_study(params, trnX, trnY, tstX, tstY, d_max)
    print(f"DT overfitting study saved for {params['params'][0]}")

if __name__ == "__main__":
    run_trees_study(d_max=25, lag=2)