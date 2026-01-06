import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from data_loader import load_data

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

df = load_data()
target_col = 'Total'

df_hourly = df[target_col].resample('h').sum()

print("Computing Decomposition...")
decomposition = seasonal_decompose(df_hourly, model='add')

fig = decomposition.plot()
fig.set_size_inches(14, 10)
fig.suptitle('Seasonal Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Lab5/traffic/results/4_seasonal_decomposition.png')
plt.close()
