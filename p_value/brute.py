"""
P-value computation with brute force
"""

import numpy as np
from result import Result
from tqdm import tqdm

def brute_ts_vals(test_statistic, transform, n_dim, n=50000):
    """
    Return list of TS vales from brute force MC
    """
    cube = np.random.rand(n, n_dim)
    pseudo_data = transform(cube)
    return np.array([test_statistic(pd) for pd in tqdm(pseudo_data)])


def brute(test_statistic, transform, n_dim, observed, n=50000):
    """
    Brute force MC sampling for p-value
    """
    ts_vals = brute_ts_vals(test_statistic, transform, n_dim, n)

    p_value = sum(ts_vals > observed) / n

    p_value_uncertainty = (p_value * (1. - p_value) / n)**0.5
    return Result(p_value, p_value_uncertainty, n)
