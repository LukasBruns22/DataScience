import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data

df = load_data()
target_col = 'Total'
output_dir = "Lab5/traffic/results"

print("\n" + "="*40)
print("       DIMENSIONALITY REPORT")
print("="*40)

if not df.empty:
    # A. Dataset Structure
    n_obs, n_vars = df.shape
    print(f"\n1. DATASET STRUCTURE")
    print(f"   - Number of Observations (Rows): {n_obs}")
    print(f"   - Number of Variables (Cols):    {n_vars}")
    print(f"   - Columns: {', '.join(df.columns)}")

    # B. Temporal Dimension
    start_date = df.index.min()
    end_date = df.index.max()
    time_span = end_date - start_date
    
    print(f"\n2. TEMPORAL DIMENSION")
    print(f"   - Start Date:     {start_date}")
    print(f"   - End Date:       {end_date}")
    print(f"   - Total Duration: {time_span}")

    FREQ = '15min' 
    df = df.asfreq(FREQ)
    missing_vals = df[target_col].isnull().sum()
    missing_pct = (missing_vals / n_obs) * 100
    
    print(f"\n3. TARGET COMPLETENESS ('{target_col}')")
    print(f"   - Missing Values: {missing_vals}")
    print(f"   - Missing %:      {missing_pct:.2f}%")
    
else:
    print("DataFrame is empty. Please check your CSV file.")

# D. Visualization of Dimensionality
if not df.empty:
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[target_col], linewidth=0.7)
    plt.title(f'Dimensionality Visualization: Full Time Series Extent ({target_col})', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/1_dimensionality.png")
    plt.close()

