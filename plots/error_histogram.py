"""
Verify correctness of error estimates
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm, chi2

import definitions

sys.path.append("./../examples")
from nested_errors import get_mn_pval_single, get_brute_mc_pval


# Settings

nd = 5
targ = 19.8
n_live = 100
significance = norm.isf(chi2.sf(targ, df=nd))
pkl_name = "data/err_check.pkl"
seed = 125
repeats = 1000

# Generate data

try:
    with open(pkl_name, 'rb') as pkl:
        log10_p, log10_p_err, nevals, log10_p_mc, log10_p_err_mc = pickle.load(pkl)
except IOError:
    log10_p, log10_p_err, nevals = [], [], []
    for i in tqdm(range(repeats)):
        seed += i  # Simply incremeent seed. Is this safe? Correlations?
        # reduce sampling efficiency to remove bias from ellipsoidal sampling
        a = get_mn_pval_single(i, file_root="./data/mn", n_dim=nd, target_obs=targ, save_all=False, seed=seed, sampling_efficiency=0.01)
        log10_p.append(a[0])
        log10_p_err.append(a[1])
        nevals.append(a[2])

    brute_mc_res = get_brute_mc_pval(1e4, n_dim=nd, target_obs=targ)
    log10_p_mc = brute_mc_res[0]
    log10_p_err_mc = brute_mc_res[1]

    with open(pkl_name, 'wb') as pkl:
        pickle.dump((log10_p, log10_p_err, nevals, log10_p_mc, log10_p_err_mc), pkl)


mean_log10_p, std_log10_p = np.mean(log10_p), np.std(log10_p)
mean_log10_p_err = np.mean(log10_p_err)

p_expected = chi2.sf(targ, df=nd)
log10_p_expected = np.log10(p_expected)
log10_p_err_expected = np.sqrt(-np.log(p_expected) / n_live) / np.log(10.)

print("log10(p) = {:.4f} +/- {:.4f} (avg MN error: {:.4f}); expected: {:.4f} +/- {:.4f}; mc: {:.4f} +/- {:.4f}.".format(mean_log10_p, std_log10_p, mean_log10_p_err, log10_p_expected, log10_p_err_expected, log10_p_mc, log10_p_err_mc))

mean_nevals, std_nevals = np.mean(nevals), np.std(nevals)
print("Required {:} +/- {:} functions evals.".format(mean_nevals, std_nevals))

# Check difference between theory and observed
error_on_mean = np.std(log10_p) / len(log10_p)**0.5
z = (mean_log10_p - log10_p_expected) / error_on_mean
print("Disrepancy between mean and expected = ", z)

# Plot data

definitions.set_style()
fig, ax = plt.subplots(figsize=(3.4, 2.25))

lpvals = np.linspace(mean_log10_p - 5. * std_log10_p, mean_log10_p + 5. * std_log10_p, 150)
gaussian_fit = norm.pdf(lpvals, loc=mean_log10_p, scale=std_log10_p)
expected = norm.pdf(lpvals, loc=log10_p_expected, scale=log10_p_err_expected)

ax.hist(log10_p, bins=15, fc='b', ec='none', alpha=0.25, density=True, label="NS results")
ax.plot(lpvals, gaussian_fit, 'b-', label="Gaussian fit")
ax.plot(lpvals, expected, 'r--', label="Expected")

ax.set_xlabel(r"$\log_{10} p$")
ax.set_ylabel("PDF")
ax.set_xlim([-3.4, -2.2])
ax.set_yticks([])

# Legend with custom order
handles, labels = ax.get_legend_handles_labels()
order = [2, 0, 1]
ohandles = [handles[o] for o in order]
olabels = [labels[o] for o in order]
leg = plt.legend(ohandles, olabels, frameon=False, handlelength=1.9, fontsize=7)
leg.set_title("${:d}d$ Gaussian\nTarget $\chi^2$ = {:.1f}".format(nd, targ), prop={'size': 7})

plt.tight_layout()
plt.savefig("error_check.pdf")
