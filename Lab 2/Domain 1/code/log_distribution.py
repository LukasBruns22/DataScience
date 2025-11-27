import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator

INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv'
OUTPUT_DIR = 'graphs/lab2/traffic_accidents/distribution/log_scale'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

def plot_log_injuries():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    casualty_vars = [
        'injuries_total', 'injuries_fatal', 'injuries_incapacitating', 
        'injuries_non_incapacitating', 'injuries_reported_not_evident',
        'injuries_no_indication', 'num_units'
    ]
    
    print(f"--- Re-plotting {len(casualty_vars)} Variables (Clean Log Histograms) ---")

    for col in casualty_vars:
        if col not in df.columns: continue
        
        data = df[col].dropna().sort_values()
        
        # --- SMART BINNING ---
        # Forces bars to align with integers (0, 1, 2) so they look solid, not like thin lines
        unique_vals = data.nunique()
        if unique_vals < 50:
            max_val = int(data.max())
            bins = np.arange(0, max_val + 2) - 0.5
        else:
            bins = 30

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Clean Histogram: Log Scale, Raw Counts (No Density normalization)
        ax.hist(data, bins=bins, color='#2c3e50', alpha=0.85, log=True, edgecolor='white')
        
        ax.set_title(f"Distribution of {col} (Log Scale)", fontsize=14)
        ax.set_xlabel(f"{col} (Count)", fontsize=12)
        ax.set_ylabel("Frequency (Log Scale)", fontsize=12)
        
        # Force integer ticks on X-axis for cleaner look
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        safe_name = col.replace('/', '_')
        plt.savefig(f"{OUTPUT_DIR}/{safe_name}_log.png")
        plt.close()
        print(f"   -> Saved {safe_name}_log.png")

if __name__ == "__main__":
    plot_log_injuries()