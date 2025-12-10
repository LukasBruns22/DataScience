import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf

from data_loader import load_data

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (16, 6)

df = load_data()
target_col = 'Inflation Rate (%)'

series_orig = df[target_col]                  
series_yearly = df[target_col].resample('YE').mean() # Yearly
series_quaterly = df[target_col].resample('QE').mean()  # Quarterly

granularities = {
    'O': series_orig,
    'Q': series_quaterly,
    'Y': series_yearly
}

print("--- ANALYSE DE DISTRIBUTION ---")


# 5-NUMBER SUMMARY 

print("\n=== 1. 5-Number Summary ===")

summary_df = pd.DataFrame()
for name, series in granularities.items():
    summary_df[name] = series.describe()

print(summary_df)


# BOXPLOTS 

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Boxplots: Detection of Outliers and Spread', fontsize=16)

for i, (name, series) in enumerate(granularities.items()):
    sns.boxplot(y=series, ax=axes[i], color='lightblue')
    axes[i].set_title(name)
    axes[i].set_ylabel('Inflation Rate (%)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Lab5/eco/results/3_boxplots.png')
plt.close()


# 4. HISTOGRAMS

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Variable Distribution: Histograms & KDE', fontsize=16)

for i, (name, series) in enumerate(granularities.items()):
    sns.histplot(series, kde=True, ax=axes[i], color='teal', bins=30)
    axes[i].set_title(name)
    axes[i].set_xlabel('Inflation Rate (%)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Lab5/eco/results/3_histograms.png')
plt.close()

# LAG PLOTS 

lag = series_orig.to_frame(name='Original')
lag['lag1M'] = lag['Original'].shift(1)
lag['lag6M'] = lag['Original'].shift(6)
lag['lag1A'] = lag['Original'].shift(12)  

plt.figure(figsize=(15, 5))
plt.plot(lag.index, lag['Original'], 
         label='Original', color='#008fd5', linewidth=2.5, alpha=0.9)
plt.plot(lag.index, lag['lag1M'], 
         label='lag1M', color='#e5247e', linewidth=2, alpha=0.8)
plt.plot(lag.index, lag['lag6M'], 
         label='lag6M', color='#e5ae38', linewidth=2, alpha=0.8)
plt.plot(lag.index, lag['lag1A'], 
         label='lag1A', color="#17823b", linewidth=2, alpha=0.8)

plt.title('Time Series Lag Overlay: Original vs Past Values', fontsize=14)
plt.ylabel('Inflation Rate (%)')
plt.xlabel('Date')
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=1)
plt.margins(x=0)

plt.tight_layout()
plt.savefig('Lab5/eco/results/3_lag.png')
plt.close()

# LAG AUTOCORRELATION PLOTS 

lags_to_show = range(1, 9) 
n_lags = len(lags_to_show)

for i, (name, series) in enumerate(granularities.items()):
    fig1 = plt.figure(figsize=(20, 8))
    fig1.suptitle(f'Lag Plots and Autocorrelation: {name}', fontsize=16)
    gs = fig1.add_gridspec(2, n_lags, height_ratios=[1, 1])

    for i, lag in enumerate(lags_to_show):
        ax = fig1.add_subplot(gs[0, i])

        lag_plot(series, lag=lag, ax=ax, s=10, alpha=0.6, c='#008fd5')

        ax.set_title(f'Lag {lag}', fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Original')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([]) 

    ax_acf = fig1.add_subplot(gs[1, :])

    plot_acf(series.dropna(), ax=ax_acf, lags=10, alpha=None, title='')

    ax_acf.set_title("Autocorrelation", fontsize=12, color='#008fd5')
    ax_acf.set_xlabel("Lags")
    ax_acf.set_ylabel("Correlation")
    for line in ax_acf.lines:
        line.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f'Lab5/eco/results/3_lag_acf_{name}.png')
    plt.close()

