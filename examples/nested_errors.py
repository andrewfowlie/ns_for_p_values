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


def get_mn_pval(task_id, n_batch_size, n_dim=4, target_obs=9.0):
   res = []
   obs = chi2.isf(chi2.sf(3.0**2, df=1), df=n_dim)
   for i in range(n_batch_size):
      temp, _ = mn(test_statistic, transform, n_dim, target_obs, n_live=100, basename='mn_{:d}'.format(int(task_id)), resume=False, ev_data=False)
      res.append( np.array([temp.error_log10_pvalue(), temp.log10_pvalue()]) )
   return np.array(res)
