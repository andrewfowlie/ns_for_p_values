import sys
import numpy as np

from scipy.stats import poisson

sys.path.append("/usr/users/hoof1/ns_for_p_values/p_value")
sys.path.append("/usr/users/hoof1/ns_for_p_values/examples")
import higgs_functions as higgs
from ns import pc

targ = float(sys.argv[1])
print("Staring run. Target: {}.".format(targ), flush=True)
n_dim_higgs = len(higgs.expected_bkg)

def transform(cube):
    return poisson.ppf(cube, mu=higgs.expected_bkg)

# r1, r2 = ns.mn(higgs.nested_ts, transform, n_dim_higgs, targ, n_live=100, basename="chains/mn_", resume=False, ev_data=True)
r1, r2 = pc(higgs.nested_ts, transform, n_dim_higgs, targ, n_live=100, file_root="pc_higgs", feedback=2, resume=True, ev_data=True)

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma)".format(targ, 10**r1.log10_pvalue, r1.significance))
