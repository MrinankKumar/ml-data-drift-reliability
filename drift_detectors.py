from scipy.stats import ks_2samp

def ks_drift(x_ref, x_new):
    score, _ = ks_2samp(x_ref, x_new)
    return score

import numpy as np

def psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    expected_counts = np.histogram(expected, bins=expected_perc)[0]
    actual_counts = np.histogram(actual, bins=expected_perc)[0]

    expected_ratio = expected_counts / len(expected)
    actual_ratio = actual_counts / len(actual)

    psi_value = np.sum((expected_ratio - actual_ratio) *
                       np.log((expected_ratio + 1e-8) / (actual_ratio + 1e-8)))
    return psi_value

def mean_shift(x_ref, x_new):
    return abs(x_ref.mean() - x_new.mean())
