import pandas as pd
import numpy as np
import category_encoders as ce  # pip install category_encoders
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FILE = 'datasets/Combined_flights_2022/flights_no_leakage.csv'
OUTPUT_FILE = 'datasets/Combined_flights_2022/Combined_Flights_2022_FuLl_Prepared.csv'

# Variables for Target Encoding (Symbolic)
TARGET_COLS = ['Hub_Airline','Route', 'Airline', 'Origin', 'Dest']

# Variables for Cyclical Encoding (Numeric-Time)
# We define the max value for the cycle (e.g., 2400 for time, 12 for months)
CYCLICAL_VARS = {
    'Month': 12,
    'DayOfWeek': 7,
    'DayofMonth': 31,
    'CRSDepTime': 2400,
    'CRSArrTime': 2400
}

def encode_cyclical(df, col, max_val):
    """Creates Sin/Cos features and drops the original column."""
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df.drop(columns=[col])

print("--- STEP 1: VARIABLE ENCODING ---")

# 1. LOAD DATA
print(f"1. Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# 2. CREATE THE ROUTE VARIABLE (Before Splitting)
print("   Generating 'Route' variable (Origin_Dest)...")
df['Route'] = df['Origin'].astype(str) + "_" + df['Dest'].astype(str)

# B. Hub_Airline (Airline @ Origin) <--- NEW!
df['Hub_Airline'] = df['Airline'].astype(str) + "_" + df['Origin'].astype(str)

# Ensure Target is numeric (0/1) for Target Encoding to work
df['Cancelled'] = df['Cancelled'].astype(int)

# 2. SPLIT DATA (Crucial Step!)
# We must split now so that our Target Encoding doesn't "cheat" by seeing the test data.
# We will recombine them at the end.
X = df.drop(columns=['Cancelled'])
y = df['Cancelled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"   Data Split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. APPLY TARGET ENCODING
print("2. Applying Target Encoding (Airline, Origin, Dest)...")
# Smoothing prevents overfitting on rare airports
encoder = ce.TargetEncoder(cols=TARGET_COLS, smoothing=10.0)

# FIT on Train, TRANSFORM both
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# 4. APPLY CYCLICAL ENCODING
print("3. Applying Cyclical Encoding (Dates & Times)...")
for col, max_val in CYCLICAL_VARS.items():
    # Only encode if the column exists
    if col in X_train_encoded.columns:
        X_train_encoded = encode_cyclical(X_train_encoded, col, max_val)
        X_test_encoded = encode_cyclical(X_test_encoded, col, max_val)

# 5. RECOMBINE & SAVE
print(f"4. Saving processed dataset to {OUTPUT_FILE}...")

# Reattach targets
X_train_encoded['Cancelled'] = y_train
X_test_encoded['Cancelled'] = y_test

# Concatenate back into one file for the next step (Imputation)
# Note: We keep the index so you can technically re-split later if needed, 
# but usually we just reset it for a clean CSV.
final_df = pd.concat([X_train_encoded, X_test_encoded])

final_df.to_csv(OUTPUT_FILE, index=False)
print("   DONE! File saved.")
print(f"   New column count: {final_df.shape[1]}")