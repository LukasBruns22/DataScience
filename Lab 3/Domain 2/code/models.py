# models.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def get_models() -> Dict[str, Any]:
    """
    Return a dict of baseline models for each step:
    - Naive Bayes
    - KNN
    """
    models = {
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5), 
    }
    return models

def train_and_evaluate_models(
    X_train, y_train, X_test, y_test
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Train NB and KNN on the given dataset and return:
      - metrics per model
      - confusion matrices per model
    """
    models = get_models()
    metrics_per_model = {}
    conf_mats = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics_per_model[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        conf_mats[name] = cm

    return metrics_per_model, conf_mats
