import pandas as pd
import os

INPUT_FILE = 'datasets/traffic_accidents/traffic_accidents.csv'

if os.path.exists(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE)
    
    # .any() checks if there is at least one True value
    # We chain it twice: once for columns, once for the whole dataframe
    has_any_nan = df.isnull().values.any()
    
    print(f"Is there at least one single explicit NaN in the file? {has_any_nan}")
    
    if has_any_nan:
        total_count = df.isnull().sum().sum()
        print(f"Total count of NaN cells: {total_count}")
        print("\nColumns containing these NaNs:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
else:
    print(f"File {INPUT_FILE} not found.")