"""
P-value computation with brute force
"""

import numpy as np
from .result import Result

def brute(test_statistic, transform, n_dim, observed, n=50000):
    """
    Brute force MC sampling for p-value 
    """
    cube = np.random.rand(n, n_dim)
    pseudo_data = transform(cube)

    p_value = 0.
    for p in pseudo_data:
        if test_statistic(p) > observed:
            p_value += 1. / n

    p_value_uncertainty = (p_value * (1. - p_value) / n)**0.5
    return Result(p_value, p_value_uncertainty, n)
