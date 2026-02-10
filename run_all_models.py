import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import get_models
from drift_simulation import apply_drift  
from drift_detection import compute_drift_score


def run_experiment():

    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = get_models()
    drift_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {}

    for model_name, model in models.items():
        results[model_name] = []

        for drift in drift_levels:
            X_train_drifted, X_test_drifted = apply_drift(X_train, X_test, drift_level=drift)
            drift_score = compute_drift_score(X_train, X_test_drifted)
            print(f"Drift Score: {drift_score}")
            drift_scores.append(drift_score)



            model.fit(X_train_drifted, y_train)
            y_pred = model.predict(X_test_drifted)

            acc = accuracy_score(y_test, y_pred)
            results[model_name].append(acc)

    for model_name, acc_list in results.items():
        plt.plot(drift_levels, acc_list, label=model_name)

    plt.xlabel("Drift Level")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison under Data Drift")
    plt.legend()

    plt.savefig("plots/model_comparison_under_drift.png")
    plt.close()

    return drift_levels, results