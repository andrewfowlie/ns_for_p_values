import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

sys.path.append("./../p_value")
import higgs_functions as higgs
import ns

targ = 9.2
n_dim_higgs = len(higgs.expected_bkg)

def transform(cube):
    return poisson.ppf(cube, mu=higgs.expected_bkg)

# r1, r2 = ns.mn(higgs.nested_ts, transform, n_dim_higgs, targ, n_live=100, basename="chains/mn_", resume=False, ev_data=True)
r1, r2 = ns.pc(higgs.nested_ts, transform, n_dim_higgs, targ, n_live=100, file_root="pc_", feedback=2, resume=False, ev_data=True)

plt.plot(r2[0], r2[1])
plt.show()

print("P-value at TS = {:.2f}: {:.2e} ({:.1f} sigma)".format(targ, 10**r1.log10_pvalue, r1.significance))
