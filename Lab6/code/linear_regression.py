from pandas import read_csv, Series
from utils import plot_forecasting_eval, plot_forecasting_series
from matplotlib.pyplot import savefig
from numpy import arange
from sklearn.linear_model import LinearRegression

train = read_csv("Datasets/traffic_forecasting/processed_train.csv", index_col="Datetime", parse_dates=True)
test = read_csv("Datasets/traffic_forecasting/processed_test.csv", index_col="Datetime", parse_dates=True)

train: Series = train["Total"]
test: Series = test["Total"]

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(train)+len(test)).reshape(-1, 1)
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Linear Regression")
savefig(f"Lab6/results/LR/evals.png")


plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"Linear Regression",
    xlabel="Datetime",
    ylabel="Traffic Volume",
)
savefig(f"Lab6/results/LR/forecast.png")