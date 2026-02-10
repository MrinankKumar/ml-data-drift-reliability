import numpy as np
from scipy.stats import ks_2samp

def compute_drift_score(X_train, X_test):
    scores = []

    for i in range(X_train.shape[1]):
        stat, _ = ks_2samp(X_train[:, i], X_test[:, i])
        scores.append(stat)

    return np.mean(scores)
