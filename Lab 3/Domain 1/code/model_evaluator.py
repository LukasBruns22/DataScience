import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='knn', approach_name='experiment_1'):
    """
    Trains a model, evaluates it, and saves the confusion matrix.
    """
    
    # Ensure plots folder exists
    os.makedirs('Lab 3/Domain 1/plots', exist_ok=True)

    print(f"\n=== Training {model_type.upper()} | Approach: {approach_name} ===")

    # 1. Initialize Model
    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'nb':
        model = GaussianNB()
    else:
        raise ValueError("model_type must be 'knn' or 'nb'")

    # 2. Train (Fit)
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_test)

    # 4. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') 
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    # print("\n--- Classification Report ---")
    # print(classification_report(y_test, y_pred))

    # 5. Generate & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Handle class labels for display
    try:
        labels = model.classes_
    except AttributeError:
        labels = np.unique(y_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix - {model_type.upper()} - {approach_name}")
    
    # Save file
    filename = f"cm_{model_type}_{approach_name}.png"
    plt.savefig(f"Lab 3/Domain 1/plots/{filename}")
    plt.close() 
    # print(f"Confusion matrix saved as '{filename}'")

    # Return metrics for comparison
    return {
        'approach': approach_name,
        'model': model_type,
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix_file': filename
    }

def compare_strategies_averaged(results_list, metric='f1_score'):
    """
    Groups results by 'approach', calculates the MEAN score,
    and identifies the best data preprocessing strategy.
    
    NOTE: Modified to ALWAYS return the winner, ignoring any 'baseline' check.
    """
    print("\n" + "-"*40)
    print(f"   STEP COMPARISON (Based on {metric})")
    print("-"*40)
    
    # Convert list of dicts to DataFrame for easy grouping
    df_results = pd.DataFrame(results_list)
    
    if df_results.empty:
        print("No results to compare.")
        return None, 0.0

    # Since we are now running separate tournaments, we just take the max score directly.
    # If there were multiple runs (folds), we would group by mean.
    summary = df_results.groupby('approach')[['accuracy', 'f1_score']].mean()
    summary = summary.sort_values(by=metric, ascending=False)
    
    print(f"\nRanking:\n{summary}")
    
    best_approach = summary.index[0]
    best_score = summary.iloc[0][metric]

    print(f"\n>>> WINNER: '{best_approach}' with {metric}={best_score:.4f}")
    
    # Always return the best approach and its score
    return best_approach, best_score