# config.py

RANDOM_STATE = 42
TARGET_COL = "Cancelled"

# Paths (adjust to your environment)
RAW_DATA_PATH = "path_to_file/Combined_Flights_2022.csv"
STEP1_AND_2_OUTPUT_PATH = "../data/step1_and_2_encoding_and_imputing_best.parquet"
STEP3_OUTPUT_PATH = "../data/step3_outliers_best.parquet"
STEP4_OUTPUT_PATH = "../data/step4_scaling_best.parquet"
STEP5_OUTPUT_PATH = "../data/step5_balancing_best.parquet"
STEP6_OUTPUT_PATH = "../data/step6_feature_selection_best.parquet"
STEP7_OUTPUT_PATH = "../data/step7_feature_generation_best.parquet"

# Where to save PDFs per step
REPORTS_DIR = "reports"
