import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv'
OUTPUT_DIR = 'graphs/lab2/traffic_accidents/sparsity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

def run_full_target_analysis():
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE)
    target = 'crash_type'
    
    # 1. DYNAMICALLY SELECT ALL NUMERIC VARIABLES
    # This grabs every single column that is a number (int or float)
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns that are likely IDs (if any) or the target itself if it were numeric
    numeric_vars = [c for c in numeric_vars if 'id' not in c.lower()]
    
    print(f"Variables included in analysis ({len(numeric_vars)}): {numeric_vars}")
    
    # Filter Data
    df_subset = df[numeric_vars + [target]].dropna()
    
    # SAMPLE DATA (Crucial!)
    # Plotting 200,000 points in a 10x10 grid will crash Python or create a black blob.
    # 1000 points is standard for visualizing density/sparsity.
    df_sample = df_subset.sample(n=1000, random_state=42)

    # --- CHART 1: FULL SPARSITY PAIRPLOT ---
    print("Generating Full Sparsity Matrix...")
    plt.figure(figsize=(20, 20)) # Large size for all variables
    
    chart = sns.pairplot(
        df_sample, 
        vars=numeric_vars, # Use ALL numeric variables
        hue=target,        # Color by Crash Type
        palette='viridis',
        diag_kind='hist',
        plot_kws={'alpha': 0.5, 's': 15}
    )
    
    chart.fig.suptitle(f"Sparsity Study: All Variables per Class ({target})", y=1.02, fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/sparsity_per_class_full.png")
    plt.close()
    print("Saved: sparsity_per_class_full.png")

    # --- CHART 2: FULL CORRELATION MATRIX ---
    print("Generating Full Correlation Heatmap...")
    
    # Encode Target to Numeric for Correlation
    df_corr = df_subset.copy()
    df_corr['TARGET_encoded'] = pd.factorize(df_corr[target])[0]
    
    # Compute Correlation
    corr_matrix = df_corr.corr().abs()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='Blues', 
        vmin=0, vmax=1,
        linewidths=0.5
    )
    plt.title(f"Correlation Analysis (All Variables + Target)", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/correlation_with_target_full.png")
    plt.close()
    print("Saved: correlation_with_target_full.png")

if __name__ == "__main__":
    run_full_target_analysis()