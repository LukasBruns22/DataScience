from sklearn.base import RegressorMixin
from pandas import Series, read_csv, DataFrame
from utils import series_train_test_split, plot_forecasting_eval, plot_forecasting_series
import matplotlib.pyplot as plt


class SimpleAvgRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean: float = 0.0
        return

    def fit(self, X: Series):
        self.mean = X.mean()
        return

    def predict(self, X: Series) -> Series:
        prd: list = len(X) * [self.mean]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series

train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime")
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime")

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Simple Average")
plt.savefig(f"Lab6/results/simple_average/evals.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Simple Average",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
plt.savefig(f"Lab6/results/simple_average/plot_forecast.png")
