from sklearn.base import RegressorMixin
from pandas import Series, read_csv, DataFrame
from utils import plot_forecasting_eval, plot_forecasting_series
import matplotlib.pyplot as plt

class PersistenceOptimistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last: float = 0.0
        return

    def fit(self, X: Series):
        self.last = X.iloc[-1]
        # print(self.last)
        return

    def predict(self, X: Series):
        prd: list = X.shift().values.ravel()
        prd[0] = self.last
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
    
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

train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime")
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime")

fr_mod = PersistenceOptimistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"One-Set Behind Persistence")
plt.savefig(f"Lab6/results/Persistence/one_set_evals.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"One-Set Behind Persistence",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
plt.savefig(f"Lab6/results/Persistence/one_set_forecast.png")

fr_mod = PersistenceRealistRegressor()
fr_mod.fit(train)
prd_trn: Series = fr_mod.predict(train)
prd_tst: Series = fr_mod.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Long Term Persistence")
plt.savefig(f"Lab6/results/Persistence/long_term_evals.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Long Term Persistence",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
plt.savefig(f"Lab6/results/Persistence/long_term_forecast.png")