import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pipeline import run_step_comparison

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score

def auto_select_and_plot_redundancy(X_train, X_test, y_train, y_test, dataset_name="Dataset", thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.7, 1.0, 15)
    
    plot_thresholds = []
    plot_n_features = []
    plot_avg_scores = []
    full_results = []

    print(f"--- Starting Redundancy Analysis for {dataset_name} ---")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    print(f"{'Threshold':<10} | {'Feats':<6} | {'KNN B.Acc':<10} | {'NB B.Acc':<10} | {'Avg B.Acc'}")
    print("-" * 65)

    # 2. Boucle de test des seuils
    for t in thresholds:
        to_drop = [column for column in upper.columns if any(upper[column] > t)]
        X_tr_red = X_train.drop(columns=to_drop)
        X_te_red = X_test.drop(columns=to_drop)
        n_feats = X_tr_red.shape[1]
        
        knn = KNeighborsClassifier()
        knn.fit(X_tr_red, y_train)
        knn_acc = balanced_accuracy_score(y_test, knn.predict(X_te_red))
        
        nb = GaussianNB()
        nb.fit(X_tr_red, y_train)
        nb_acc = balanced_accuracy_score(y_test, nb.predict(X_te_red))
 
        avg_acc = (knn_acc + nb_acc) / 2
    
        plot_thresholds.append(t)
        plot_n_features.append(n_feats)
        plot_avg_scores.append(avg_acc)
        full_results.append({
            'threshold': t,
            'n_features': n_feats,
            'avg_score': avg_acc,
            'dropped_cols': to_drop
        })
        
        print(f"{t:<10.2f} | {n_feats:<6} | {knn_acc:<10.4f} | {nb_acc:<10.4f} | {avg_acc:.4f}")

    best_config = max(full_results, key=lambda x: (x['avg_score'], -x['n_features']))
    best_t = best_config['threshold']
    
    print(f"\n---> Best Threshold selected: {best_t:.2f} (Avg B.Acc: {best_config['avg_score']:.4f})")
    print(f"---> Features reduced from {X_train.shape[1]} to {best_config['n_features']}")

    # 4. Génération du Graphique
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axe Y gauche : Performance (Rouge)
    color_score = 'tab:red'
    ax1.set_xlabel('Correlation Threshold')
    ax1.set_ylabel('Avg Balanced Accuracy (KNN+NB)', color=color_score, fontweight='bold')
    ax1.plot(plot_thresholds, plot_avg_scores, color=color_score, marker='o', linewidth=2, label='Avg B.Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_score)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Axe Y droit : Nombre de features (Bleu)
    ax2 = ax1.twinx()
    color_feat = 'tab:blue'
    ax2.set_ylabel('Number of Features Remaining', color=color_feat, fontweight='bold')
    ax2.plot(plot_thresholds, plot_n_features, color=color_feat, marker='s', linestyle='--', label='Num Features')
    ax2.tick_params(axis='y', labelcolor=color_feat)

    # Marquer le meilleur seuil choisi
    ax1.axvline(x=best_t, color='green', linestyle=':', linewidth=2)
    ax1.annotate(f'Selected: {best_t:.2f}', xy=(best_t, best_config['avg_score']), 
                 xytext=(best_t-0.05, best_config['avg_score']-0.05),
                 arrowprops=dict(facecolor='green', shrink=0.05), color='green', fontweight='bold')

    plt.title(f'Redundancy Selection: Performance vs. Feature Count ({dataset_name})')
    fig.tight_layout()
    plt.savefig('ds_latex/images/2_d1_redundancy.png')
    
    # 5. Application finale aux datasets
    X_train_final = X_train.drop(columns=best_config['dropped_cols'])
    X_test_final = X_test.drop(columns=best_config['dropped_cols'])
    
    return X_train_final, X_test_final, best_t, fig

# --- APPROACH 1: Keep More Features (Chi-Squared) ---
def app1_chi2(X_train, y_train, X_test, y_test):
    print("      (Selecting Top Features using Chi2...)")
    
    # Clip negatives for Chi2
    X_train_clip = X_train.clip(lower=0)
    X_test_clip = X_test.clip(lower=0)
    
    # Select top 50% of features
    k = max(15, int(X_train.shape[1] * 0.5))
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train_clip, y_train)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> Selected {len(cols)} features (from {X_train.shape[1]})")
    
    # Transform
    X_train_sel = X_train[cols]
    X_test_sel = X_test[cols]
    
    return X_train_sel, y_train, X_test_sel, y_test

# --- APPROACH 2: Random Forest ---
def app2_random_forest(X_train, y_train, X_test, y_test):
    print("      (Selecting features using Random Forest importance...)")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Keep features with importance > median
    threshold = np.median(importances)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    
    # Get columns
    cols = X_train.columns[selector.get_support()]
    print(f"      -> RF kept {len(cols)} features (from {X_train.shape[1]})")
    
    # If we kept too few, just keep top 50%
    if len(cols) < X_train.shape[1] * 0.3:
        print(f"      -> Too few features, keeping top 50% by importance instead")
        indices = np.argsort(importances)[::-1][:int(X_train.shape[1] * 0.5)]
        cols = X_train.columns[indices]
    
    X_train_sel = X_train[cols]
    X_test_sel = X_test[cols]
    
    return X_train_sel, y_train, X_test_sel, y_test

def run_feature_selection_step(X_train, y_train, X_test, y_test):
    X_train, X_test, best_t, fig = auto_select_and_plot_redundancy(X_train, X_test, y_train, y_test, "Traffic Accidents")
    return run_step_comparison(
        X_train, y_train, X_test, y_test,
        app1_func=app1_chi2,
        app2_func=app2_random_forest,
        step_name="FEATURE SELECTION"
    )