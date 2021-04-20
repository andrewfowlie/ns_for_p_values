"""
Functions for quick runs of NS on toy problem to investigate errors
===================================================================
"""

import sys
import os
import numpy as np
from scipy.special import ndtri

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../p_value/')
from ns import mn
from brute import brute


def transform(cube):
    return ndtri(cube)

def test_statistic(observed_):
    return (observed_**2).sum()

def get_mn_pval_single(unique_id, file_root="./mn", n_dim=2, target_obs=9.0, n_live=100, save_all=False, **kwargs):
    bname = file_root
    if save_all:
        bname += "_{:d}".format(int(unique_id))
    temp = mn(test_statistic, transform, n_dim, target_obs, n_live=n_live, basename=bname, resume=False, ev_data=False, **kwargs)
    return np.array([temp.log10_pvalue, temp.error_log10_pvalue, temp.calls])

def get_brute_mc_pval(n_brute_samples=1e6, n_dim=2, target_obs=9.0, **kwargs):
    temp = brute(test_statistic, transform, n_dim, target_obs, n=int(n_brute_samples), **kwargs)
    return np.array([temp.log10_pvalue, temp.error_log10_pvalue])
