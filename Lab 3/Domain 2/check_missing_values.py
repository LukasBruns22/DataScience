import pandas as pd

# 1. Load your encoded dataset
file_path = 'datasets/Combined_flights_2022/flight_data_encoded.csv'

print(f"Reading {file_path}...")
df = pd.read_csv(file_path)

# 2. Count missing values per column
missing_counts = df.isnull().sum()

# Filter to show only columns that actually have missing data
missing_data = missing_counts[missing_counts > 0]

print("\n--- MISSING VALUES REPORT ---")
if missing_data.empty:
    print("✅ No missing values found!")
    print("Action: You can SKIP the Imputation Step.")
    print("Next Step: Outliers Treatment.")
else:
    print("⚠️  Found missing values in the following columns:")
    print(missing_data)
    print(f"\nTotal rows with missing data: {df.isnull().any(axis=1).sum()}")
    print("Action: You MUST proceed to compare two Imputation approaches.")