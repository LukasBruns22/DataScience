import pandas as pd

def remove_data_leakage(df):
    # This function expects 'df' to be a DataFrame, not a string path!
    injury_cols = [col for col in df.columns if 'injur' in col.lower()]
    other_leakage_cols = ['damage', 'prim_contributory_cause']
    cols_to_drop = injury_cols + other_leakage_cols
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    print(f"Removing {len(cols_to_drop)} leakage columns: {cols_to_drop}")
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean

# --- FIX IS HERE ---

# 1. LOAD the file first
# (Make sure this path is correct relative to where you run the script)
df = pd.read_csv('Lab 3/Domain 1/code/traffic_accidents.csv') 

# 2. PASS the loaded dataframe to the function
df_cleaned = remove_data_leakage(df)

# 3. SAVE the result
df_cleaned.to_csv('traffic_accidents_cleaned.csv', index=False)

print("\nSuccess! Remaining columns:")
print(df_cleaned.columns.tolist())