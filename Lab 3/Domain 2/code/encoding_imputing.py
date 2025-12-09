import pandas as pd
import numpy as np

from config import STEP1_AND_2_OUTPUT_PATH, TARGET_COL, RANDOM_STATE
from data_utils import load_raw_data, save_step_data, train_test_split_xy
from models import train_and_evaluate_models

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

STEP_NAME = "step_1_encoding_and_missing"


# -------------------------------------------------------------------
# Method One – Cyclical encoding for time blocks
# -------------------------------------------------------------------
def convert_block_column(df, col):
    print(f"[convert_block_column] Converting {col}...")

    if col not in df.columns:
        print(f"  Column {col} not in df. Skipping.")
        return df

    df = df.copy()
    df[col] = df[col].astype(str)

    def parse_block(block):
        try:
            s, e = block.split("-")
            sh, sm = int(s[:2]), int(s[2:])
            eh, em = int(e[:2]), int(e[2:])
            start = sh * 60 + sm
            end = eh * 60 + em
            return (start + end) / 2
        except Exception:
            return np.nan

    minutes = df[col].apply(parse_block)

    df[col + "_sin"] = np.sin(2 * np.pi * minutes / 1440)
    df[col + "_cos"] = np.cos(2 * np.pi * minutes / 1440)

    print(f"  Created {col}_sin and {col}_cos")

    return df.drop(columns=[col])


# -------------------------------------------------------------------
# CRS Time cyclical encoding
# -------------------------------------------------------------------
def convert_crs_times(df):
    df = df.copy()

    def encode(series, name):
        minutes = series.fillna(0).astype(int).apply(
            lambda x: (x // 100) * 60 + (x % 100)
        )
        df[name + "_sin"] = np.sin(2 * np.pi * minutes / 1440)
        df[name + "_cos"] = np.cos(2 * np.pi * minutes / 1440)

    if "CRSDepTime" in df.columns:
        encode(df["CRSDepTime"], "CRSDepTime")
    else:
        print("[convert_crs_times] CRSDepTime missing")

    if "CRSArrTime" in df.columns:
        encode(df["CRSArrTime"], "CRSArrTime")
    else:
        print("[convert_crs_times] CRSArrTime missing")

    return df


# -------------------------------------------------------------------
# Extract columns **after** the cyclic transform
# -------------------------------------------------------------------
def get_encoding_A_cols(df, max_unique_for_oh=20):
    print("[get_encoding_A_cols] Determining categorical vs numeric columns...")

    print("  Columns at this stage:", df.columns.tolist())

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)

    low_card = [c for c in cat_cols if df[c].nunique() <= max_unique_for_oh]
    high_card = [c for c in cat_cols if df[c].nunique() > max_unique_for_oh]

    print(f"  Numeric columns: {len(num_cols)}")
    print(f"  Categorical (low-card): {low_card}")
    print(f"  Categorical (high-card, DROPPED): {high_card}")

    return num_cols, low_card


# -------------------------------------------------------------------
# Build the preprocessing pipeline
# -------------------------------------------------------------------
def build_preprocessor(df, imputation):
    print(f"\n[build_preprocessor] Building preprocessor with imputation='{imputation}'")

    # Simulate preprocessing for column detection
    print("  -> Running temp cyclical transform for column inference...")
    df_tmp = convert_crs_times(convert_block_column(convert_block_column(df.copy(), "DepTimeBlk"), "ArrTimeBlk"))

    print("  Columns after all cyclical transforms:", df_tmp.columns.tolist())

    num_cols, cat_cols = get_encoding_A_cols(df_tmp)

    print(f"  -> num_cols={num_cols}")
    print(f"  -> cat_cols={cat_cols}")

    block_transformer = FunctionTransformer(
        lambda X: convert_crs_times(
            convert_block_column(
                convert_block_column(X.copy(), "DepTimeBlk"), "ArrTimeBlk"
            )
        ),
        validate=False
    )

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    if imputation == "simple":
        print("  -> Using SIMPLE imputation")
        ct = ColumnTransformer([
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ])

        preprocessor = Pipeline([
            ("block", block_transformer),
            ("ct", ct),
        ])

    elif imputation == "knn":
        print("  -> Using KNN imputation")
        ct_encode = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", categorical_pipe, cat_cols),
        ])

        preprocessor = Pipeline([
            ("block", block_transformer),
            ("encode", ct_encode),
            ("knn_imputer", KNNImputer(n_neighbors=5)),
        ])

    else:
        raise ValueError("imputation must be simple or knn")

    print("[build_preprocessor] Preprocessor ready.")
    return preprocessor, num_cols, cat_cols


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("\n===== STEP 1: Load Raw Data =====")
    df_raw = load_raw_data()
    print(f"Loaded dataset with shape: {df_raw.shape}")
    print("Columns BEFORE leakage drop:", df_raw.columns.tolist())

    # Drop leakage columns
    leakage_cols = [
        "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
        "TaxiOut", "TaxiIn", "WheelsOff", "WheelsOn", "ArrTime", "ArrDelay",
        "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups", "Tail_Number",
        "ActualElapsedTime", "AirTime", "Diverted", "DivAirportLandings", "Duplicate",
    ]

    print("\n===== Dropping Leakage Columns =====")
    df_raw = df_raw.drop(columns=[c for c in leakage_cols if c in df_raw.columns])
    print("Columns AFTER leakage drop:", df_raw.columns.tolist())

    print("\n===== Train/Test Split =====")
    X_train, X_test, y_train, y_test = train_test_split_xy(df_raw)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    N_SAMPLES = 200_000
    if len(X_train) > N_SAMPLES:
        print(f"[Downsampling] Reducing training set from {len(X_train)} → {N_SAMPLES}")
        sample_idx = np.random.choice(X_train.index, size=N_SAMPLES, replace=False)
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
    else:
        print(f"[Downsampling] Training size < {N_SAMPLES}, skipping.")

    combos = {
        "EncA_Simple": "simple",
        "EncA_KNN": "knn",
    }

    results = {}
    preprocessors = {}

    print("\n===== Evaluating Encoding Options =====")
    for name, imp in combos.items():
        print(f"\n--- {name} ---")
        preprocessor, _, _ = build_preprocessor(df_raw, imp)

        print("  -> Fitting preprocessor...")
        X_train_t = preprocessor.fit_transform(X_train)
        print(f"  -> X_train transformed shape: {X_train_t.shape}")

        X_test_t = preprocessor.transform(X_test)
        print(f"  -> X_test transformed shape: {X_test_t.shape}")

    print("  -> Training models...")
    metrics, _ = train_and_evaluate_models(X_train_t, y_train, X_test_t, y_test)

    best_f1 = max(m["f1"] for m in metrics.values())
    print(f"  -> best f1 = {best_f1}")

    results[name] = best_f1
    preprocessors[name] = preprocessor

    # Pick best
    print("\n===== Selecting Best Preprocessor =====")
    best_name = max(results, key=lambda k: results[k])
    best_preprocessor = preprocessors[best_name]
    print(f"BEST: {best_name} with f1={results[best_name]}")

    print("\n===== Transforming Full Dataset =====")
    X_full = df_raw.drop(columns=[TARGET_COL])
    y_full = df_raw[TARGET_COL]

    X_full_trans = best_preprocessor.fit_transform(X_full)
    print(f"  -> Full transformed shape: {X_full_trans.shape}")

    print("\n===== Building Feature Names =====")
    df_tmp = convert_crs_times(convert_block_column(convert_block_column(df_raw.copy(), "DepTimeBlk"), "ArrTimeBlk"))
    num_cols, cat_cols = get_encoding_A_cols(df_tmp)

    feature_names = num_cols.copy()

    ct = (
        best_preprocessor.named_steps["ct"]
        if "ct" in best_preprocessor.named_steps
        else best_preprocessor.named_steps["encode"]
    )
    ohe = ct.named_transformers_["cat"].named_steps["onehot"]
    ohe_names = ohe.get_feature_names_out(cat_cols)
    feature_names.extend(ohe_names)

    print(f"Total features: {len(feature_names)}")
    print("Feature list ", feature_names)

    X_full_df = pd.DataFrame(X_full_trans, columns=feature_names)
    X_full_df[TARGET_COL] = y_full.values

    print("\n===== Saving Output =====")
    save_step_data(X_full_df, STEP1_AND_2_OUTPUT_PATH)
    print("Saved.")


if __name__ == "__main__":
    main()
