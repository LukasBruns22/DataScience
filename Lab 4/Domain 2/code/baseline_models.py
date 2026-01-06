import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
DATASET_PATH = "datasets/Combined_Flights_2022.csv"
OUTPUT_DIR = "graphs/BaselineModels"
TARGET = "Cancelled"
FILE_TAG = "Combined_Flights_2022"

def load_and_preprocess_data():
    print("Loading data...")
    # Load raw dataset
    df = pd.read_csv(DATASET_PATH)
    
    # 1. Handle Target
    # Cancelled is usually boolean (False/True) in this dataset, convert to int (0/1)
    if df[TARGET].dtype == bool:
        df[TARGET] = df[TARGET].astype(int)
    
    # 2. Remove Leakage Columns (Columns that are only known after the flight/cancellation)
    # DepDelay, ArrDelay, Taxies, Wheels, AirTime, ActualElapsedTime, Diverted etc.
    leakage_candidates = [
        "ActualElapsedTime", "AirTime", "ArrDel15", "ArrDelay", "ArrDelayMinutes", "ArrTime", "ArrTimeBlk", 
        "ArrivalDelayGroups", "DOT_ID_Marketing_Airline", "DOT_ID_Operating_Airline", "DepDel15", "DepDelay", 
        "DepDelayMinutes", "DepTime", "DepTimeBlk", "DepartureDelayGroups", "DestAirportID", "DestAirportSeqID", 
        "DestCityMarketID", "DestCityName", "DestState", "DestStateFips", "DestStateName", "DestWac", 
        "DistanceGroup", "DivAirportLandings", "Diverted", "FlightDate", "Flight_Number_Marketing_Airline", 
        "Flight_Number_Operating_Airline", "IATA_Code_Marketing_Airline", "IATA_Code_Operating_Airline", 
        "Marketing_Airline_Network", "Operated_or_Branded_Code_Share_Partners", "Operating_Airline", 
        "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID", "OriginCityName", "OriginState", 
        "OriginStateFips", "OriginStateName", "OriginWac", "Quarter", "Tail_Number", "TaxiIn", "TaxiOut", 
        "WheelsOff", "WheelsOn", "Year", "CancellationCode", "CarrierDelay", "WeatherDelay", "NASDelay", 
        "SecurityDelay", "LateAircraftDelay"
    ]
    # Filter to only existing columns
    cols_to_drop = [c for c in leakage_candidates if c in df.columns]
    print(f"Dropping leakage columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 3. Select Numeric Features Only (for baseline simplicity)
    # Keep Target
    target_data = df[TARGET]
    df = df.select_dtypes(include=[np.number])
    df[TARGET] = target_data # Ensure target is kept
    
    # 4. Drop Missing Values
    df.dropna(inplace=True)
    
    return df

def train_and_evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_and_preprocess_data()
    
    # Split data
    y = df[TARGET].values
    X = df.drop(columns=[TARGET]).values
    vars = df.drop(columns=[TARGET]).columns.tolist()
    
    # Split 70/30
    trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # ---------------------------------------------------------
    # SAMPLING LOGIC (Undersampling Majority Class)
    # ---------------------------------------------------------
    trnY = np.array(trnY)
    n_neg = len(np.where(trnY == 0)[0])
    n_pos = len(np.where(trnY == 1)[0])
    print(f"Original distribution: 0={n_neg}, 1={n_pos}")
    
    # We undersample the majority class (0) to match the minority class (1)
    neg_indices = np.where(trnY == 0)[0]
    pos_indices = np.where(trnY == 1)[0]
    
    np.random.seed(42) # Ensure reproducibility
    sampled_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
    
    balanced_indices = np.concatenate([sampled_neg_indices, pos_indices])
    np.random.shuffle(balanced_indices)
    
    trnX = trnX[balanced_indices]
    trnY = trnY[balanced_indices]
    
    print(f"Sampled distribution: {np.unique(trnY, return_counts=True)}")
    # ---------------------------------------------------------
    
    labels = np.unique(y)
    labels.sort()
    
    # Models config with User Specified Parameters
    models = [
        {
            'name': 'Naive Bayes',
            'model': GaussianNB(),
            'params': {'name': 'NB', 'metric': 'f1', 'params': ('GaussianNB')} 
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42),
            'params': {'name': 'DT', 'metric': 'f1', 'params': ('entropy', 2)}
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=0.1, random_state=42),
            'params': {'name': 'GB', 'metric': 'f1', 'params': (500, 2, 0.1)}
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
            'model': MLPClassifier(learning_rate='adaptive', learning_rate_init=0.005, max_iter=500, random_state=42),
            'params': {'name': 'MLP', 'metric': 'f1', 'params': ('adaptive', 0.005, 500)}
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(max_depth=2, max_features=0.7, n_estimators=750, random_state=42),
            'params': {'name': 'RF', 'metric': 'f1', 'params': (2, 0.7, 750)}
        }
    ]
    
    for m in models:
        print(f"Training {m['name']}...")
        model = m['model']
        model.fit(trnX, trnY)
        
        prd_trn = model.predict(trnX)
        prd_tst = model.predict(tstX)
        
        # Plot evaluation results
        plt.figure()
        plot_evaluation_results(m['params'], trnY, prd_trn, tstY, prd_tst, labels)
        
        # Save figure
        # Removing spaces from model name for filename
        clean_name = m['name'].replace(" ", "")
        filename = f"{OUTPUT_DIR}/{FILE_TAG}_{clean_name}_baseline_eval.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    train_and_evaluate()
