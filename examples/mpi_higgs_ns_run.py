import sys
import os
import numpy as np

from scipy.stats import poisson

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../examples/')
sys.path.insert(0, script_dir+'/../p_value/')
import higgs_functions as higgs
from ns import pc

targ = float(sys.argv[1])
res = bool(sys.argv[2])
root = str(sys.argv[3])
print("Staring run. Target: {}.".format(targ), flush=True)
n_dim_higgs = len(higgs.expected_bkg)

def transform(cube):
    return poisson.ppf(cube, mu=higgs.expected_bkg)

#r1, r2 = pc(higgs.nested_ts_simple, transform, n_dim_higgs, targ, n_live=100, file_root=root, feedback=2, resume=res, ev_data=True, do_clustering=False)
r1, r2 = pc(higgs.nested_ts_simple_fast, transform, n_dim_higgs, targ, n_live=100, file_root=root, feedback=2, resume=res, ev_data=True, do_clustering=False)

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma), {} calls.".format(targ, 10**r1.log10_pvalue, r1.significance, r1.calls))
