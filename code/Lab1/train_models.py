import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# --- SETUP PLOTTING STYLE ---
sns.set(style="whitegrid")
GRAPH_DIR = os.path.join("graphs", "lab1")
os.makedirs(GRAPH_DIR, exist_ok=True)

# 1. CONFIGURATION
# 1. CONFIGURATION - OPTIMIZED FOR LAB 1
model_params = {
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': np.logspace(-10, -8, 5) # Keep this, it's fast anyway
        },
        'plot_param': 'var_smoothing'
    },
    
    'Logistic Regression': {
        # Solver 'liblinear' is fast and supports both l1 and l2
        'model': LogisticRegression(max_iter=500, solver='liblinear'), 
        'params': {
            'C': [0.01, 0.1, 1, 10, 100], 
            'penalty': ['l2', 'l1']
        },
        'plot_param': 'C'
    },
    
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 9, 15], 
            'weights': ['uniform'], 
            'metric': ['euclidean', 'manhattan']
        },
        'plot_param': 'n_neighbors'
    },
    
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [3, 5, 10, 20], 
            'min_samples_split': [5, 10],
            'criterion': ['gini'] 
        },
        'plot_param': 'max_depth'
    },
    
    'MLP (Neural Net)': {
        'model': MLPClassifier(max_iter=200, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)], 
            'activation': ['relu'],
            'alpha': [0.0001, 0.01]
        },
        'plot_param': 'hidden_layer_sizes'
    }
}

def plot_small_multiples(clf, model_name, dataset_name):
    results = clf.cv_results_
    param_grid = clf.param_grid
    num_params = len(param_grid)

    cols = 2
    rows = int(np.ceil(num_params / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for idx, (param_name, values) in enumerate(param_grid.items()):
        ax = axes[idx]

        param_values = [str(p[param_name]) for p in results['params']]
        acc = results['mean_test_Accuracy']
        prec = results['mean_test_Precision']
        rec = results['mean_test_Recall']

        ax.plot(param_values, acc, marker='o', label='Acc')
        ax.plot(param_values, prec, marker='s', label='Prec')
        ax.plot(param_values, rec, marker='^', label='Rec')
        ax.set_title(param_name)
        ax.set_xticklabels(param_values, rotation=45)
        ax.grid(True)
        ax.legend()

    # Hide empty subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{model_name} – Hyperparameter Grid Performance", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    filename = f"{dataset_name}_{model_name.replace(' ', '_')}_small_multiples.png"
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()


def plot_hyperparameter_performance(clf, model_name, param_name, dataset_name):
    """
    Plots Validation Accuracy, Precision, and Recall vs the hyperparameter.
    """
    if param_name is None:
        return

    # Extract results from GridSearch
    results = clf.cv_results_
    params = results['params']
    
    # Extract the specific parameter values for the X-axis
    param_values = [str(p[param_name]) for p in params]
    
    # Extract mean scores for each metric
    mean_acc = results['mean_test_Accuracy']
    mean_prec = results['mean_test_Precision']
    mean_rec = results['mean_test_Recall']
    
    plt.figure(figsize=(10, 6))
    
    # Plot 3 lines
    plt.plot(param_values, mean_acc, marker='o', label='Accuracy', color='blue', linewidth=2)
    plt.plot(param_values, mean_prec, marker='s', label='Precision', color='green', linestyle='--', linewidth=2)
    plt.plot(param_values, mean_rec, marker='^', label='Recall', color='red', linestyle=':', linewidth=2)
    
    plt.title(f'{model_name}: Performance vs {param_name}', fontsize=15)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Rotate x-labels if they are long (like MLP tuples)
    if model_name == "MLP (Neural Net)":
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    
    # Save file
    filename = f"{dataset_name}_{model_name.replace(' ', '_')}_hyperparams.png"
    save_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Graph saved: {save_path}")

def plot_final_comparison(results_df, dataset_name):
    """
    Creates a grouped bar chart comparing the BEST version of all models.
    """
    df_melted = results_df.melt(id_vars="Model", 
                                value_vars=["Accuracy", "Precision", "Recall"],
                                var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="viridis")
    
    plt.title("Final Model Comparison (Best Config)\n(Hyperparameters shown in CSV)", fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(GRAPH_DIR, "{dataset_name}_model_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\n-> Final Comparison Graph saved: {save_path}")

def run_training(file_path, target_col, dataset_name):
    print(f"\n{'='*40}\nProcessing Dataset: {file_path}\n{'='*40}")
    
    # Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return
    
    # --- TARGET VARIABLE SETUP ---
    
    if target_col not in df.columns:
        print(f"Target '{target_col}' not found. Columns available: {df.columns.tolist()}")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    scoring_metrics = {
        'Accuracy': 'accuracy', 
        'Precision': 'precision_weighted', 
        'Recall': 'recall_weighted'
    }

    # Loop through models
    for model_name, mp in model_params.items():
        print(f"Training {model_name}...")
        
        clf = GridSearchCV(mp['model'], mp['params'], cv=3, 
                           scoring=scoring_metrics, refit='Accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # --- PLOTTING STEP ---
        plot_small_multiples(clf, model_name, dataset_name)
        plot_hyperparameter_performance(clf, model_name, mp['plot_param'], dataset_name)
        
        # Evaluate Best Model
        best_model = clf.best_estimator_
        best_params = clf.best_params_

        y_pred = best_model.predict(X_test)
        
        # Final Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Best_Params': best_params
        })
        
    # Final Results & Comparison Plot
    results_df = pd.DataFrame(results)
    print("\nFINAL RESULTS:\n", results_df)
    
    # Save CSV
    results_df.to_csv(f'results_summary_{dataset_name}.csv', index=False)
    
    # Generate Comparison Plot
    plot_final_comparison(results_df, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate multiple classification models on a cleaned dataset."
    )
    parser.add_argument(
        '-p', '--path', 
        type=str, 
        required=True, 
        help='Path to the input CSV file to be trained (e.g., datasets/file.csv)'
    )

    parser.add_argument(
        '-n', '--name', 
        type=str, 
        required=True, 
        help='Name of the dataset'
    )

    parser.add_argument(
        '-t', '--target', 
        type=str, 
        required=True, 
        help='Name of the target variable/column (e.g., Cancelled)'
    )    

    args = parser.parse_args()

    run_training(args.path, args.target, args.name)