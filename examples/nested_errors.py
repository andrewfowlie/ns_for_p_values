import sys
import os
import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../p_value/')
from ns import mn


def transform(cube):
    return ndtri(cube)

def test_statistic(observed_):
    return (observed_**2).sum()

def get_mn_pval_single(unique_id, file_root="./mn", n_dim=2, target_obs=9.0):
   obs = chi2.isf(chi2.sf(3.0**2, df=1), df=n_dim)
   temp = mn(test_statistic, transform, n_dim, target_obs, n_live=100, basename=file_root+"_{:d}".format(int(unique_id)), resume=False, ev_data=False)
   return np.array([temp.error_log10_pvalue, temp.log10_pvalue])

def get_mn_pval(task_id, n_batch_size, n_dim=2, target_obs=9.0):
   res = []
   for i in range(n_batch_size):
      print("Running batch {} in task {}".format(i, task_id))
      res.append( get_mn_pval_single(task_id*n_batch_size+i, "./mn", n_dim, target_obs) )
   return np.array(res)
