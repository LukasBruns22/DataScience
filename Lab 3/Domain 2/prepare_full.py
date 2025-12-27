import pandas as pd
from sklearn.model_selection import train_test_split

# --- IMPORTS: THE WINNING APPROACHES ---
from feature_generation import app2_context_bins as step_feature_generation
from outliers import app1_isolation_forest as step_outliers
from scaling import app2_standard as step_scaling
from feature_selection import app1_chi2 as step_feature_selection
from balancing import app2_undersample as step_balancing 

# --- CONFIGURATION ---
START_FILE = 'datasets/Combined_Flights_2022/Combined_Flights_2022_Full_Encode.csv' 
FINAL_FILE = 'datasets/Combined_Flights_2022/flight_data_prepared_full.csv'
TARGET_COL = 'Cancelled'

def main():
    print("=======================================================")
    print("   STARTING PREPARATION PIPELINE (Single Tagged File)")
    print("=======================================================")
    
    # 1. Load Data
    print(f"1. Loading Full Dataset...")
    df = pd.read_csv(START_FILE)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 2. Train/Test Split
    print("2. Splitting Data (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 3. Apply Steps (Transforming the data)
    
    # A: Feature Generation
    X_train, y_train, X_test, y_test = step_feature_generation(X_train, y_train, X_test, y_test)
    
    # B: Outliers (Removes rows from Train only)
    X_train, y_train, X_test, y_test = step_outliers(X_train, y_train, X_test, y_test)
    
    # C: Scaling
    X_train, y_train, X_test, y_test = step_scaling(X_train, y_train, X_test, y_test)
    
    # D: Feature Selection
    X_train, y_train, X_test, y_test = step_feature_selection(X_train, y_train, X_test, y_test)
    
    # E: Balancing (Undersamples Train only)
    X_train, y_train, X_test, y_test = step_balancing(X_train, y_train, X_test, y_test)
    
    print("   -> Pipeline Steps Complete.")

    # 4. TAG AND MERGE
    print("\n4. Tagging and Merging...")
    
    # Create Train DataFrame and add 'Set' column
    df_train_final = X_train.copy()
    df_train_final[TARGET_COL] = y_train
    df_train_final['Set'] = 'Train'  # <--- The Magic Tag
    
    # Create Test DataFrame and add 'Set' column
    df_test_final = X_test.copy()
    df_test_final[TARGET_COL] = y_test
    df_test_final['Set'] = 'Test'    # <--- The Magic Tag
    
    # Combine
    df_full = pd.concat([df_train_final, df_test_final])
    
    # 5. Save
    df_full.to_csv(FINAL_FILE, index=False)
    
    print(f"DONE! Saved to: {FINAL_FILE}")
    print(f"Total Rows: {len(df_full)}")
    print(f"   - Train rows: {len(df_train_final)}")
    print(f"   - Test rows:  {len(df_test_final)}")

if __name__ == "__main__":
    main()


# How to use this
'''
import pandas as pd

# Load the big file
df = pd.read_csv('datasets/Combined_Flights_2022/flight_data_prepared_all_in_one.csv')

# Separate them using the 'Set' column
train_data = df[df['Set'] == 'Train'].copy()
test_data = df[df['Set'] == 'Test'].copy()

# Drop the 'Set' column since models can't use it
train_data = train_data.drop(columns=['Set'])
test_data = test_data.drop(columns=['Set'])

# Define X and y
X_train = train_data.drop(columns=['Cancelled'])
y_train = train_data['Cancelled']

X_test = test_data.drop(columns=['Cancelled'])
y_test = test_data['Cancelled']

print("Ready to train!")
'''