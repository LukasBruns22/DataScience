import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv' # Make sure this matches your file name
OUTPUT_DIR = 'graphs/lab2/traffic_accidents/granularity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set(style="whitegrid")

def run_granularity_analysis():
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # ==========================================
    # 1. TEMPORAL GRANULARITY (Professor's Method)
    # ==========================================
    print("Analyzing Temporal Granularity (Year, Quarter, Month, Day, Hour)...")
    
    # Ensure date format
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    
    # Derive variables as per Professor's logic
    df['year'] = df['crash_date'].dt.year
    df['quarter'] = df['crash_date'].dt.quarter
    df['month'] = df['crash_date'].dt.month
    df['day_of_week'] = df['crash_date'].dt.dayofweek # 0=Mon, 6=Sun
    df['hour'] = df['crash_hour'] # Already exists, but renaming for consistency

    # Define the hierarchy levels
    time_levels = ['year', 'quarter', 'month', 'day_of_week', 'hour']
    
    # Create one large figure with subplots (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Temporal Granularity Analysis', fontsize=16)

    for i, level in enumerate(time_levels):
        # Count and sort by INDEX (Time), not Frequency
        counts = df[level].value_counts().sort_index()
        
        sns.barplot(x=counts.index, y=counts.values, ax=axes[i], color='#4c72b0')
        axes[i].set_title(level.capitalize())
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Records' if i==0 else '')
        
        # Clean up x-ticks for busy plots (like Year or Hour)
        if level == 'hour':
            axes[i].set_xticks([0, 6, 12, 18, 23])
            axes[i].set_xticklabels(['0', '6', '12', '18', '23'])

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/temporal_granularity_full.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    # ==========================================
    # 2. SYMBOLIC HIERARCHY: SEVERITY (Taxonomy)
    # ==========================================
    print("Analyzing Severity Hierarchy (Crash Type -> Injury)...")
    
    severity_vars = ['crash_type', 'most_severe_injury']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Property Granularity: Accident Severity', fontsize=16)
    
    for i, var in enumerate(severity_vars):
        # Sort by frequency here because it's symbolic
        counts = df[var].value_counts().nlargest(10) 
        sns.barplot(x=counts.values, y=counts.index, ax=axes[i], color='#55a868')
        axes[i].set_title(f"Level {i+1}: {var} (Granular)")
        axes[i].set_xlabel('Records')

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/severity_hierarchy.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

    # ==========================================
    # 3. HIGH CARDINALITY CHECK (Need for Taxonomy)
    # ==========================================
    print("Analyzing High Cardinality (Contributory Cause)...")
    
    col = 'prim_contributory_cause'
    plt.figure(figsize=(10, 8))
    
    # Show top 20 to demonstrate the "Long Tail"
    counts = df[col].value_counts().nlargest(20)
    sns.barplot(x=counts.values, y=counts.index, color='#c44e52')
    
    plt.title(f'Taxonomy Requirement: {col} (Top 20)', fontsize=14)
    plt.xlabel('Records')
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/cause_granularity_check.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_granularity_analysis()