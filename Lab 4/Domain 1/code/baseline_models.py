import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import os
import sys

# Ensure code directory is in path for imports if running from domain root
sys.path.append('code')
from utils.dslabs_functions import plot_evaluation_results, CLASS_EVAL_METRICS

# Constants
DATASET_PATH = "datasets/traffic_accidents.csv"
OUTPUT_DIR = "graphs/BaselineModels"
TARGET = "crash_type" # Assuming this is the target, need to verify if it needs encoding or if it's already encoded in raw?
# Wait, user said "remove data leakage columns... then Discard all non-numeric data". 
# If target is non-numeric, it might be dropped. 
# User said "traffic_accidents_prepared_train... comes from traffic_accidents... we have done all necessary data preparation... like encoding". 
# But for this task: "take datasets/traffic_accidents.csv... eliminate all non-numeric variables". 
# If the target is categorical in the raw file, and we drop all non-numeric, we lose the target. 
# I should probably Encode the target first if it's not numeric, OR assume the user implies "variables" as in features.
# Let's check the csv content first or assume standard "Crash_Type" is target.
# Actually, the user said "Discard all non-numeric data". Usually target is categorical. 
# I will check the file content first to be safe, but for now I will write the script assuming I need to encode the target if it is not numeric, 
# BEFORE dropping non-numeric features.

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)
    
    # Remove leakage columns
    leakage_cols = [
        "damage", "injuries_fatal", "injuries_incapacitating", 
        "injuries_no_indication", "injuries_non_incapacitating", 
        "injuries_reported_not_evident", "injuries_total", 
        "most_severe_injury"
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Eliminate empty variables (all NaN or empty, already covered by dropna? "All variables totally empty")
    df.dropna(axis=1, how='all', inplace=True)
    
    # "All records having some missing values" -> done by dropna()
    
    # "Discard all non-numeric data, eliminating all non-numeric variables"
    # Essential: Preserve Target. 
    # Current target in prepared files is "crash_type_enc". In raw it's likely "Crash_Type" or "CRASH_TYPE".
    # I should find the target column.
    
    # Let's try to identify target. Usually it's crash_type.
    target_col = None
    for col in df.columns:
        if col.lower() == 'crash_type':
            target_col = col
            break
            
    if target_col:
        # User said "eliminating all non-numeric variables".
        # If target IS non-numeric, we must encode it first.
        if df[target_col].dtype == 'object':
             # Simple encoding for target
             df[target_col] = df[target_col].astype('category').cat.codes
    
    # Now keep only numeric
    df = df.select_dtypes(include=[np.number])
    
    return df, target_col

def train_and_evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df, target_col = load_and_preprocess_data()
    
    if target_col not in df.columns:
        # Fallback if target was dropped or not found (unlikely if strictly following instructions, but safety check)
        # If target was 'crash_type' and it was numeric, it's there. 
        # If it was encoding above, it's there.
        # If it wasn't found, we might have an issue.
        # Let's assume the user knows what they are asking and the target ends up being numeric or is 'crash_type'.
        # For now, let's assume the last column is target if not found, or 'crash_type'.
        pass

    # Split data
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Split 70/30 standard since not specified, but usually standard.
    trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    labels = np.unique(y)
    labels.sort()
    
    # Models config
    models = [
        {
            'name': 'Naive Bayes',
            'model': BernoulliNB(),
            'params': {'name': 'NB', 'metric': 'f1', 'params': ('BernoulliNB')} 
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42),
            'params': {'name': 'DT', 'metric': 'f1', 'params': ('entropy', 10)}
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=0.3, random_state=42),
            'params': {'name': 'GB', 'metric': 'f1', 'params': (500, 2, 0.3)}
        },
        {
            'name': 'KNN',
            'model': KNeighborsClassifier(n_neighbors=31, metric='manhattan'),
            'params': {'name': 'KNN', 'metric': 'f1', 'params': ('manhattan', 31)}
        },
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=100, random_state=42),
            'params': {'name': 'LR', 'metric': 'f1', 'params': ('l1', 100)}
        },
        {
            'name': 'MLP',
            'model': MLPClassifier(learning_rate='constant', learning_rate_init=0.5, max_iter=750, random_state=42),
            'params': {'name': 'MLP', 'metric': 'f1', 'params': ('constant', 0.5, 750)}
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(max_depth=7, max_features=0.3, n_estimators=500, random_state=42),
            'params': {'name': 'RF', 'metric': 'f1', 'params': (7, 0.3, 500)}
        }
    ]
    
    for m in models:
        print(f"Training {m['name']}...")
        model = m['model']
        model.fit(trnX, trnY)
        
        prd_trn = model.predict(trnX)
        prd_tst = model.predict(tstX)
        
        # Use existing plot function
        plot_evaluation_results(m['params'], trnY, prd_trn, tstY, prd_tst, labels)
        
        # Save figure
        # Filename format as implied: "one figure per model... exact like i have done"
        # usually: code uses savefig(f'graphs/DT/{FILE_TAG}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
        # I will save to graphs/BaselineModels/dataset_model_params_eval.png
        
        filename = f"{OUTPUT_DIR}/traffic_accidents_{m['params']['name']}_baseline_eval.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    train_and_evaluate()
