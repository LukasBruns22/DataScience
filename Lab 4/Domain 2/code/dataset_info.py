# Basic but complete EDA for a dataset with target variable "Cancelled"

import pandas as pd
import numpy as np

def dataset_overview(df: pd.DataFrame, target: str = "Cancelled"):
    print("===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMN TYPES =====")
    print(df.dtypes.value_counts())
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    print("\n===== DUPLICATES =====")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df[target].value_counts())
    print("\nTarget proportion:")
    print(df[target].value_counts(normalize=True))

    print("\n===== NUMERICAL FEATURES =====")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    print(df[num_cols].describe().T)

    print("\n===== CATEGORICAL FEATURES =====")
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        print(f"\n-- {col} --")
        print("Unique values:", df[col].nunique())
        print(df[col].value_counts().head(10))

    print("\n===== CORRELATION WITH TARGET (numeric only) =====")
    if df[target].dtype != "object":
        corr = df[num_cols].corrwith(df[target]).sort_values(key=abs, ascending=False)
        print(corr)

    print("\n===== POTENTIAL LEAKAGE CHECK =====")
    leak_candidates = [c for c in df.columns if target.lower() in c.lower() and c != target]
    print("Leakage-like columns:", leak_candidates)


# Example usage
df = pd.read_csv("datasets/combined_flights_prepared_train.csv")
dataset_overview(df, target="Cancelled")