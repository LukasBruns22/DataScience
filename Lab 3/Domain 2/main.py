import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import get_model_performance
from vis import plot_pipeline_history, plot_final_confusion_matrices

from outliers import run_outliers_step
from scaling import run_scaling_step
from balancing import run_balancing_step
from feature_selection import run_feature_selection_step
from feature_generation import run_feature_generation_step

# --- CONFIGURATION ---
START_FILE = 'datasets/Combined_Flights_2022/flight_data_encoded.csv' 
FINAL_FILE = 'datasets/Combined_Flights_2022/flight_data_prepared_final.csv'

def record_history(history_list, step_name, X_train, y_train, X_test, y_test):
    """Helper to calculate current score and add to history."""
    scores = get_model_performance(X_train, y_train, X_test, y_test)
    history_list.append({
        'Step': step_name,
        'KNN': scores['knn'],
        'NB': scores['nb']
    })
    return history_list

def main():
    print("STARTING DATA PREPARATION PIPELINE (With Visualization)")
    history = [] 
    
    # 1. Load Data & Single Split
    print("1. Loading Data & Performing Single Split...")
    df = pd.read_csv(START_FILE)
    X = df.drop(columns=['Cancelled'])
    y = df['Cancelled']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Record Baseline (Raw Encoded Data)
    history = record_history(history, "Baseline", X_train, y_train, X_test, y_test)
    
    # 2. Run Steps Sequence
    
    # Step A: Feature Generation
    X_train, y_train, X_test, y_test = run_feature_generation_step(X_train, y_train, X_test, y_test)
    history = record_history(history, "Generation", X_train, y_train, X_test, y_test)
    
    # Step B: Outliers
    X_train, y_train, X_test, y_test = run_outliers_step(X_train, y_train, X_test, y_test)
    history = record_history(history, "Outliers", X_train, y_train, X_test, y_test)
    
    # Step C: Scaling
    X_train, y_train, X_test, y_test = run_scaling_step(X_train, y_train, X_test, y_test)
    history = record_history(history, "Scaling", X_train, y_train, X_test, y_test)
    
    # Step D: Feature Selection
    X_train, y_train, X_test, y_test = run_feature_selection_step(X_train, y_train, X_test, y_test)
    history = record_history(history, "Selection", X_train, y_train, X_test, y_test)
    
    # Step E: Balancing (Last)
    X_train, y_train, X_test, y_test = run_balancing_step(X_train, y_train, X_test, y_test)
    history = record_history(history, "Balancing", X_train, y_train, X_test, y_test)
    
    # 3. Final Outputs
    print("\nPipeline Complete.")
    
    # A. Save Data
    final_df = pd.concat([X_train, X_test])
    final_df['Cancelled'] = pd.concat([y_train, y_test])
    final_df.to_csv(FINAL_FILE, index=False)
    print(f"   -> Data saved to {FINAL_FILE}")
    
    # B. Generate Plots
    print("\nGenerating Performance Plots...")
    plot_pipeline_history(history)
    plot_final_confusion_matrices(X_train, y_train, X_test, y_test)
    print("DONE! Check the PNG files in your folder.")

if __name__ == "__main__":
    main()