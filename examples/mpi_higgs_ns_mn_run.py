import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../examples/')
sys.path.insert(0, script_dir+'/../p_value/')
import higgs_functions as higgs
from ns import mn

targ = float(sys.argv[1])
res = bool(sys.argv[2])
n_dim_higgs = len(higgs.expected_bkg)

print("Starting run. Target: {}.".format(targ), flush=True)

r1, r2 = mn(higgs.nested_ts_simple, higgs.generate_pseudo_data, n_dim=n_dim_higgs, observed=targ, n_live=100, basename="chains/mn_sbkg_", resume=res, ev_data=True)

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma, {} calls)".format(targ, 10**r1.log10_pvalue, r1.significance, r1.calls))
