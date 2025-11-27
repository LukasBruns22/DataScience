import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm, expon, lognorm

# --- CONFIGURATION ---
INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv' 
OUTPUT_DIR = 'graphs/lab2/traffic_accidents/distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# --- PROFESSOR'S DISTRIBUTION FITTING FUNCTION ---
def compute_known_distributions(x_values):
    distributions = dict()
    # Gaussian (Normal)
    try:
        mean, sigma = norm.fit(x_values)
        # Check for valid sigma to avoid errors
        if sigma > 0:
            distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    except: pass
    
    # Exponential
    try:
        loc, scale = expon.fit(x_values)
        distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    except: pass
    
    # LogNorm
    try:
        s, loc, scale = lognorm.fit(x_values)
        distributions["LogNor(%.1f,%.2f)" % (np.log(scale), s)] = lognorm.pdf(x_values, s, loc, scale)
    except: pass
    
    return distributions

def get_variable_types(df):
    variable_types = {"numeric": [], "binary": [], "date": [], "symbolic": []}
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            variable_types["binary"].append(c)
        else:
            try:
                pd.to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except (ValueError, TypeError):
                try:
                    pd.to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except (ValueError, TypeError):
                    variable_types["symbolic"].append(c)
    return variable_types

def run_distribution_analysis():
    print(f"\n{'='*40}\nDISTRIBUTION ANALYSIS (WITH CURVES + ALIGNED BINS)\n{'='*40}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please check your path.")
        return

    df = pd.read_csv(INPUT_FILE)
    var_types = get_variable_types(df)
    numeric_vars = var_types['numeric']

    # Filter out ID-like columns
    numeric_vars = [c for c in numeric_vars if df[c].nunique() < len(df) * 0.9]

    # --- 1. GLOBAL BOXPLOT ---
    plt.figure(figsize=(12, 6))
    df[numeric_vars].boxplot(rot=45)
    plt.title("Global Boxplot (All Numeric Variables)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/0_global_boxplot.png")
    plt.close()

    # --- 2. NUMERIC VARIABLES (Histograms + Curves) ---
    print(f"\n--- Fitting Distributions to {len(numeric_vars)} Variables ---")
    
    for col in numeric_vars:
        data_clean = df[col].dropna().sort_values()
        
        # If extremely few unique values (and not temporal), simple hist
        if data_clean.nunique() < 5:
            plt.figure(figsize=(6, 4))
            sns.histplot(data_clean, kde=False, color='skyblue')
            plt.title(f"Histogram: {col}")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/num_{col}_hist.png")
            plt.close()
            continue

        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # --- INTELLIGENT BINNING ---
        # If it's a temporal variable (integers), align bins to center on the integers
        if col in ['crash_hour', 'crash_day_of_week', 'crash_month']:
            # Create bins: [ -0.5, 0.5, 1.5, ... ] to center bars on 0, 1, 2...
            bins = np.arange(data_clean.min(), data_clean.max() + 2) - 0.5
            # Use Histogram (Density=True for curves)
            ax.hist(data_clean, bins=bins, density=True, color='skyblue', alpha=0.6, label='Data', edgecolor='white')
            # Force integer ticks on x-axis
            ax.set_xticks(np.arange(data_clean.min(), data_clean.max() + 1))
        else:
            # Standard variables
            ax.hist(data_clean, bins=20, density=True, color='skyblue', alpha=0.6, label='Data')

        # --- COMPUTE & PLOT CURVES ---
        # We pass the *values* to the fit function
        dists = compute_known_distributions(data_clean.values)
        colors = ['red', 'green', 'purple']
        
        for i, (name, y_vals) in enumerate(dists.items()):
            c = colors[i % len(colors)]
            ax.plot(data_clean, y_vals, label=name, linewidth=2, color=c)

        ax.set_title(f"Best Fit Distribution: {col}")
        ax.set_xlabel(col)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/num_{col}_best_fit.png")
        plt.close()
        print(f"Saved: num_{col}_best_fit.png")

    # --- 3. OUTLIER COMPARISON ---
    print("\n--- Generating Outlier Analysis ---")
    outlier_summary = []
    for col in numeric_vars:
        data = df[col].dropna()
        summary = data.describe()
        Q1, Q3 = summary['25%'], summary['75%']
        IQR = Q3 - Q1
        mean, std = summary['mean'], summary['std']
        
        outlier_summary.append({
            'Variable': col,
            'IQR_Outliers': ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum(),
            'StDev_3_Outliers': ((data < (mean - 3 * std)) | (data > (mean + 3 * std))).sum()
        })

    outlier_df = pd.DataFrame(outlier_summary)
    plt.figure(figsize=(12, 6))
    outlier_df.set_index('Variable')[['IQR_Outliers', 'StDev_3_Outliers']].plot(kind='bar', ax=plt.gca())
    plt.title("Outlier Detection Sensitivity (IQR vs 3-StDev)")
    plt.ylabel("Count of Outliers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/outliers_comparison.png")
    plt.close()

    # --- 4. SYMBOLIC VARS ---
    cat_vars = get_variable_types(df)['symbolic'] + get_variable_types(df)['binary']
    for col in cat_vars:
        plt.figure(figsize=(10, 6))
        top_cats = df[col].value_counts().nlargest(15)
        sns.barplot(x=top_cats.values, y=top_cats.index, color='teal')
        plt.title(f'Distribution: {col}')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cat_{col.replace('/', '_')}.png")
        plt.close()

if __name__ == "__main__":
    run_distribution_analysis()