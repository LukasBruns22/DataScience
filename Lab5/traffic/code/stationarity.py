import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from data_loader import load_data

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

df = load_data()
target_col = 'Total'

df_original = df[target_col]
df_hourly = df[target_col].resample('h').sum()
df_daily = df[target_col].resample('D').sum()

def test_stationarity(timeseries, title):
    print(f"\n" + "="*60)
    print(f"RESULTS : {title}")
    print("="*60)


    plt.figure(figsize=(14, 6))
    plt.plot(timeseries, color='#1f77b4', label='Series', linewidth=1, alpha=0.6)
    plt.plot(timeseries.index, len(timeseries)*[timeseries.mean()], color='red', label=f'Mean', linewidth=2)
    
    plt.legend(loc='best')
    plt.title(f'Stationarity Analysis : {title}')
    plt.ylabel('Volume Trafic')
    plt.savefig(f'Lab5/traffic/results/5_stationarity_{title.replace(" ", "_").lower()}.png')
    plt.close()

    print("ADF...")
    result = adfuller(timeseries)
    
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05


print("\n\n--- TEST 1 : ORIGINAL SERIE ---")
result = test_stationarity(df_original, title="Original Series")
print(f"The series {('is' if result else 'is not')} stationary")

print("\n\n--- TEST 1 : HOURLY SERIE ---")
result = test_stationarity(df_hourly, title="Hourly Series")
print(f"The series {('is' if result else 'is not')} stationary")

print("\n\n--- TEST 3 : DAILY SERIE ---")
result = test_stationarity(df_daily, title="Daily Series")
print(f"The series {('is' if result else 'is not')} stationary")