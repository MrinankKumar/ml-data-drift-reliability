import numpy as np
import pandas as pd

def mean_shift(X, drift_level):
    X_shifted = X.copy()
    shift = drift_level * np.std(X, axis=0)
    X_shifted = X_shifted + shift
    return X_shifted

def variance_shift(X, drift_level):
    X_scaled = X.copy()
    scale_factor = 1 + drift_level
    mean = np.mean(X, axis=0)
    X_scaled = mean + (X - mean) * scale_factor
    return X_scaled

def noise_injection(X, drift_level):
    X_noisy = X.copy()
    noise = np.random.normal(0, drift_level * np.std(X, axis=0), X.shape)
    X_noisy = X_noisy + noise
    return X_noisy

def feature_rotation(X, drift_level):
    theta = drift_level * np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    if X.shape[1] >= 2:
        X_rot = X.copy()
        X_rot[:, :2] = X[:, :2].dot(rotation_matrix)
        return X_rot
    return X

def apply_drift(X_train, X_test, drift_level=0.0):
    X_train_drifted = X_train.copy()
    X_test_drifted = X_test.copy()
    
    X_test_drifted = mean_shift(X_test_drifted, drift_level)
    X_test_drifted = variance_shift(X_test_drifted, drift_level)
    X_test_drifted = noise_injection(X_test_drifted, drift_level)
    X_test_drifted = feature_rotation(X_test_drifted, drift_level)
    
    return X_train_drifted, X_test_drifted

from scipy.stats import ks_2samp
import numpy as np

def compute_drift_score(X_train, X_test_drifted):
    scores = []

    for i in range(X_train.shape[1]):
        stat, _ = ks_2samp(X_train[:, i], X_test_drifted[:, i])
        scores.append(stat)

    return np.mean(scores)
