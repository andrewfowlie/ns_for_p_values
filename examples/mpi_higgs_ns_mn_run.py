import sys
import numpy as np

from scipy.stats import poisson

sys.path.append("/usr/users/hoof1/ns_for_p_values/p_value")
sys.path.append("/usr/users/hoof1/ns_for_p_values/examples")
import higgs_functions as higgs
from ns import mn

targ = float(sys.argv[1])
res = bool(sys.argv[2])
print("Staring run. Target: {}.".format(targ), flush=True)
n_dim_higgs = len(higgs.expected_bkg)

def transform(cube):
    return poisson.ppf(cube, mu=higgs.expected_bkg)

r1, r2 = mn(higgs.nested_ts_simple, transform, n_dim=n_dim_higgs, observed=targ, n_live=100, basename="chains/mn_sbkg_", resume=res, ev_data=True)

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma, {} calls)".format(targ, 10**r1.log10_pvalue, r1.significance, r1.calls))
