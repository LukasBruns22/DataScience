import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.api.types import is_numeric_dtype

# --- CONFIGURATION ---
INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv' 
OUTPUT_DIR = 'graphs/lab2/dimensionality'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

# --- PROFESSOR'S FUNCTION FOR VARIABLE TYPES ---
def get_variable_types(df):
    variable_types = {"numeric": [], "binary": [], "date": [], "symbolic": []}
    
    # Loop through all columns
    for c in df.columns:
        # 1. Check for Binary (Exactly 2 unique values, ignoring NaNs)
        if df[c].nunique(dropna=True) == 2:
            variable_types["binary"].append(c)
        else:
            try:
                # 2. Check for Numeric
                pd.to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except (ValueError, TypeError):
                try:
                    # 3. Check for Date (Only if it's not numeric)
                    # We use a strict check to avoid false positives
                    pd.to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except (ValueError, TypeError):
                    # 4. Default to Symbolic (Text/Category)
                    variable_types["symbolic"].append(c)
                    
    return variable_types

def run_dimensionality_analysis():
    print(f"\n{'='*40}\nDIMENSIONALITY & INTEGRITY ANALYSIS\n{'='*40}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Load data (Not setting na_values yet, as we want to detect 'UNKNOWN' manually)
    df = pd.read_csv(INPUT_FILE)

    # --- 1. BASIC SHAPE ---
    rows, cols = df.shape
    print(f"Records (Rows):    {rows}")
    print(f"Variables (Cols):  {cols}")
    
    # Graph 1: Rows vs Columns (Log Scale)
    plt.figure(figsize=(6, 5))
    values = {"Records": rows, "Variables": cols}
    plt.bar(values.keys(), values.values(), color=['#34495e', '#2ecc71'])
    plt.yscale('log') # Log scale to see both bars
    plt.title("Nr of records vs Nr variables")
    plt.ylabel("Count (Log Scale)")
    
    # Add text labels
    for i, v in enumerate(values.values()):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/0_records_variables.png")
    plt.close()
    print(f"Graph Saved: {OUTPUT_DIR}/0_records_variables.png")
    
    # --- 2. VARIABLE TYPES (UPDATED WITH PROF LOGIC) ---
    print("\n--- Inferring Variable Types ---")
    var_types = get_variable_types(df)
    
    counts = {k: len(v) for k, v in var_types.items()}
    print(counts)
    print(f"Binary variables found: {var_types['binary']}")

    # Graph 2: Variable Types
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), color='teal')
    plt.title("Nr of variables per type")
    plt.ylabel("Nr variables")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_variable_types.png")
    plt.close()
    print(f"Graph Saved: {OUTPUT_DIR}/1_variable_types.png")

    # --- 3. MISSING VALUES (Explicit NaN + Implicit Unknown) ---
    
    # Explicit
    explicit_missing = df.isnull().sum()
    explicit_missing = explicit_missing[explicit_missing > 0]

    # Implicit (Our "Smart" addition)
    MISSING_INDICATORS = ['UNKNOWN', 'UNABLE TO DETERMINE', 'NOT APPLICABLE', '?', 'MISSING']
    implicit_missing_counts = {}
    
    # Only scan symbolic/object columns for text placeholders
    # We can use the 'symbolic' list we just generated!
    check_cols = var_types['symbolic'] + var_types['binary']
    
    for col in check_cols:
        if col in df.columns:
            total_implicit = 0
            for indicator in MISSING_INDICATORS:
                count = df[col].astype(str).str.upper().eq(indicator).sum()
                total_implicit += count
            if total_implicit > 0:
                implicit_missing_counts[col] = total_implicit

    # Combine for the graph
    implicit_series = pd.Series(implicit_missing_counts)
    
    total_missing = pd.DataFrame({
        'Explicit (NaN)': explicit_missing,
        'Implicit (Unknown)': implicit_series
    }).fillna(0)
    
    total_missing['Total'] = total_missing.sum(axis=1)
    total_missing = total_missing.sort_values('Total', ascending=False).drop('Total', axis=1)

    if not total_missing.empty:
        print("\n--- Missing Values Detected ---")
        print(total_missing.head(10))
        
        # Graph 3: Missing Values
        plt.figure(figsize=(14, 8))
        total_missing.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'], ax=plt.gca())
        plt.title("Nr of missing values per variable")
        plt.ylabel("Nr missing values")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/2_missing_values.png")
        plt.close()
        print(f"Graph Saved: {OUTPUT_DIR}/2_missing_values.png")
    else:
        print("No missing values found.")

if __name__ == "__main__":
    run_dimensionality_analysis()