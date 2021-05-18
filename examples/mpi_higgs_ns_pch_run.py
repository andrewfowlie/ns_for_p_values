import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../examples/')
sys.path.insert(0, script_dir+'/../p_value/')
import higgs_functions as higgs
from ns import pc

targ = float(sys.argv[1])
res = bool(sys.argv[2])
basedir = str(sys.argv[3])+'/chains/'
root = str(sys.argv[4])
n_dim_higgs = len(higgs.expected_bkg)

print("Starting run. Target: {}.".format(targ), flush=True)

kwargs = { 'base_dir': basedir, 'do_clustering': False }
r1, r2 = pc(higgs.nested_ts_simple_fast, higgs.generate_pseudo_data, n_dim_higgs, targ, n_live=100, file_root=root, feedback=2, resume=res, ev_data=True, **kwargs)

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma), {} calls.".format(targ, 10**r1.log10_pvalue, r1.significance, r1.calls))
