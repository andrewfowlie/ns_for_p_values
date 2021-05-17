"""
P-value computation with brute force
"""

import numpy as np
from tqdm import tqdm

from .result import Result


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


def brute_low_memory(test_statistic, transform, n_dim, observed, n=50000):
    """
    Brute force MC sampling for p-value
    """
    p_value = np.zeros_like(observed)
    for _ in tqdm(range(n)):
        cube = np.random.rand(n_dim)
        pseudo_data = transform(cube)
        ts = test_statistic(pseudo_data)
        p_value[ts >= observed] += 1. / n

    return p_value
