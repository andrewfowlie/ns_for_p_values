import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

sys.path.append("..")
from p_value.brute import *
import p_value.ns as ns


def transform(cube):
    return norm.ppf(cube)

def test_statistic(observed_):
    return (observed_**2).sum()

def analytic_p_value(observed_, n_dim):
    return chi2.sf(observed_, df=n_dim)

def fudge_factor(n_dim, z_score=5):
    return chi2.isf(chi2.sf(z_score**2, df=1), df=n_dim)/z_score**2

def safe_log10_scal(x):
    return np.log10(x) if x > 0 else np.nan

safe_log10 = np.vectorize(safe_log10_scal)

#dimensions = [1, 2, 5, 10, 20, 50]
dimensions = [1, 2, 5, 10] # 5-d ~ 1.5 min, 10-d ~ 2.5 min, 20-d > 2 hrs(?)
obs_values = np.linspace(0, 25, num=50, endpoint=True)

n_brute_samples = int(1e6)


# Plot the p-values for a given TS using the analytical, brute-force MC, and nested sampling methods
# This plot illustrates different ways of plotting the results
fig, ax = plt.subplots()

for d in dimensions:
    t1 = time.time()
    new_obs_vals = obs_values*fudge_factor(d) # Get high enough TS values for 5 sigma
    # Analytical method
    p1 = [analytic_p_value(o, d) for o in new_obs_vals]
    ax.plot(new_obs_vals, np.log10(p1), label='$n = {:d}$'.format(d))
    # Brute-force MC. N.B. sort the results, then iterate through them. This is much, much faster
    # than using sum(ts > new_obs_vals[i]) / n_brute_samples etc. again and again.
    ts_vals = np.sort(brute_ts_vals(test_statistic, transform, n_dim=d, n=n_brute_samples))
    n_obs_vals = len(new_obs_vals)
    p2 = np.zeros(n_obs_vals)
    i = 0
    for j,ts in enumerate(ts_vals):
        if (ts > new_obs_vals[i] and i < n_obs_vals):
            p2[i] = 1. - (j+1.)/n_brute_samples
            i += 1
    ax.step(new_obs_vals, safe_log10(p2), where='post', color='k', ls='-')
    # Nested sampling method
    _, p3 = ns.mn(test_statistic, transform, d, max(new_obs_vals), n_live=250, ev_data=True)
    ax.plot(p3[:,0], np.log10(p3[:,1]), c='grey', ls='-')
    up = np.log10(p3[:,1]) + safe_log10(p3[:,2]/p3[:,1])
    lo = np.log10(p3[:,1]) - safe_log10(p3[:,2]/p3[:,1])
    ax.fill_between(p3[:,0], lo, up, alpha=0.2, ec='none', fc='grey')
    t2 = time.time()
    dt = t2 - t1
    print('Calculations for {:d} dimension(s) took {:d} mins and {:d} seconds.'.format(d, int(dt/60.), int(dt%60)))

ax.legend(frameon=False)
ax.set_ylim([-7,0])
ax.set_xlabel('TS')
ax.set_ylabel('log10(p-value)')
plt.savefig("example_multi_gaussian.pdf")
plt.show()
