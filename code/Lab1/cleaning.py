import pandas as pd
import os

# --- CONFIGURATION ---
# CHANGE THESE PATHS TO MATCH YOUR NEW DATASET
INPUT_FILE = 'datasets/Combined_Flights_2022/Combined_Flights_2022.csv'        # <--- Put your new file name here
OUTPUT_FILE = 'datasets/Combined_Flights_2022_cleaned.csv'    # <--- Where to save the cleaned version
TARGET_VAR = 'Cancelled'                            # <--- The new target variable

# 1. Load the data
if not os.path.exists(INPUT_FILE):
    print(f"Error: File not found at {INPUT_FILE}")
else:
    df = pd.read_csv(INPUT_FILE)
    print(f"Original Shape: {df.shape}")

    # 2. Drop variables (columns) that are totally empty
    df = df.dropna(axis=1, how='all')

    # 3. Drop records (rows) having ANY missing values
    df = df.dropna(axis=0, how='any')

    # 4. Encode the Target 
    # We convert 'Cancelled' to numbers (e.g., Yes=1, No=0) so it survives the cleaning
    if TARGET_VAR in df.columns:
        df[TARGET_VAR] = df[TARGET_VAR].astype('category').cat.codes
        print(f"Target '{TARGET_VAR}' encoded to numeric.")
    else:
        print(f"⚠️ WARNING: Target variable '{TARGET_VAR}' not found in dataset!")

    # 5. Discard all non-numeric data
    # Now that 'Cancelled' is a number, it will be kept.
    df_numeric = df.select_dtypes(include=['number'])

    # 6. Save the clean file
    df_numeric.to_csv(OUTPUT_FILE, index=False)

    # Final Check
    print("-" * 30)
    print("Processing complete.")
    print(f"Cleaned Shape:  {df_numeric.shape}")
    print(f"Columns kept:   {df_numeric.columns.tolist()}")
    print(f"File saved to:  {OUTPUT_FILE}")