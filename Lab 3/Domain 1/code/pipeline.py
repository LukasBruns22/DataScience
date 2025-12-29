import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, f1_score

def get_model_performance(X_train, y_train, X_test, y_test):
    """
    Trains KNN and NB and returns Balanced Accuracy, Accuracy, Recall, and F1.
    """
    # 1. Naive Bayes
    nb = GaussianNB()
    try:
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)
        nb_score = balanced_accuracy_score(y_test, nb_pred)
        nb_accuracy = accuracy_score(y_test, nb_pred)
        nb_recall = recall_score(y_test, nb_pred, zero_division=0)
        nb_f1 = f1_score(y_test, nb_pred, zero_division=0)
    except Exception:
        nb_score = nb_accuracy = nb_recall = nb_f1 = 0.0

    # 2. KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_score = balanced_accuracy_score(y_test, knn_pred)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_recall = recall_score(y_test, knn_pred, zero_division=0)
    knn_f1 = f1_score(y_test, knn_pred, zero_division=0)
    
    return {
        'nb': {
            'balanced_accuracy': nb_score,
            'accuracy': nb_accuracy,
            'recall': nb_recall,
            'f1': nb_f1
        },
        'knn': {
            'balanced_accuracy': knn_score,
            'accuracy': knn_accuracy,
            'recall': knn_recall,
            'f1': knn_f1
        }
    }

def run_step_comparison(X_train, y_train, X_test, y_test, app1_func, app2_func, step_name):
    """
    Receives FIXED Train/Test sets.
    Applies transformations, compares performance, returns the best Train/Test sets.
    """
    print(f"\n=======================================================")
    print(f"RUNNING STEP: {step_name}")
    print(f"=======================================================")
    
    # A. BASELINE
    print("   Evaluating Baseline (Input from previous step)...")
    base_res = get_model_performance(X_train, y_train, X_test, y_test)
    base_avg = (base_res['nb']['balanced_accuracy'] + base_res['knn']['balanced_accuracy']) / 2
    print(f"      -> Baseline Avg: {base_avg:.4f}")
    print(f"         NB  - Accuracy: {base_res['nb']['accuracy']:.4f}, Recall: {base_res['nb']['recall']:.4f}, F1: {base_res['nb']['f1']:.4f}, Bal.Acc: {base_res['nb']['balanced_accuracy']:.4f}")
    print(f"         KNN - Accuracy: {base_res['knn']['accuracy']:.4f}, Recall: {base_res['knn']['recall']:.4f}, F1: {base_res['knn']['f1']:.4f}, Bal.Acc: {base_res['knn']['balanced_accuracy']:.4f}")

    # B. APPROACH 1
    print("   Running Approach 1...")
    # Pass COPIES so we don't accidentally ruin the original data for App 2
    X_tr1, y_tr1, X_te1, y_te1 = app1_func(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
    
    res1 = get_model_performance(X_tr1, y_tr1, X_te1, y_te1)
    avg1 = (res1['nb']['balanced_accuracy'] + res1['knn']['balanced_accuracy']) / 2
    print(f"      -> App 1 Avg:    {avg1:.4f}")
    print(f"         NB  - Accuracy: {res1['nb']['accuracy']:.4f}, Recall: {res1['nb']['recall']:.4f}, F1: {res1['nb']['f1']:.4f}, Bal.Acc: {res1['nb']['balanced_accuracy']:.4f}")
    print(f"         KNN - Accuracy: {res1['knn']['accuracy']:.4f}, Recall: {res1['knn']['recall']:.4f}, F1: {res1['knn']['f1']:.4f}, Bal.Acc: {res1['knn']['balanced_accuracy']:.4f}")

    # C. APPROACH 2
    print("   Running Approach 2...")
    X_tr2, y_tr2, X_te2, y_te2 = app2_func(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
    
    res2 = get_model_performance(X_tr2, y_tr2, X_te2, y_te2)
    avg2 = (res2['nb']['balanced_accuracy'] + res2['knn']['balanced_accuracy']) / 2
    print(f"      -> App 2 Avg:    {avg2:.4f}")
    print(f"         NB  - Accuracy: {res2['nb']['accuracy']:.4f}, Recall: {res2['nb']['recall']:.4f}, F1: {res2['nb']['f1']:.4f}, Bal.Acc: {res2['nb']['balanced_accuracy']:.4f}")
    print(f"         KNN - Accuracy: {res2['knn']['accuracy']:.4f}, Recall: {res2['knn']['recall']:.4f}, F1: {res2['knn']['f1']:.4f}, Bal.Acc: {res2['knn']['balanced_accuracy']:.4f}")

    # D. DECISION
    print("   Deciding Winner...")
    
    # Logic: Pick the best score.
    # Note: If App 1 == Baseline, we stick with Baseline to avoid unnecessary changes.
    if avg1 > base_avg and avg1 >= avg2:
        print("      RESULT: Approach 1 is the Winner.")
        return X_tr1, y_tr1, X_te1, y_te1
        
    elif avg2 > base_avg and avg2 > avg1:
        print("      RESULT: Approach 2 is the Winner.")
        return X_tr2, y_tr2, X_te2, y_te2
        
    else:
        print("      RESULT: No Improvement. Keeping Baseline.")
        return X_train, y_train, X_test, y_test