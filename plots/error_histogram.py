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

nd = 5
targ = 19.8
n_live = 100
significance = norm.isf(chi2.sf(targ, df=nd))
pkl_name = "data/err_check.pkl"

try:
    with open(pkl_name, 'rb') as pkl:
        lgp, lgp_err, nevals, lgp_mc, lgp_err_mc = pickle.load(pkl)
except IOError:
    lgp, lgp_err, nevals = [], [], []
    for i in tqdm(range(1000)):
        a = get_mn_pval_single(i, file_root="./data/mn", n_dim=nd, target_obs=targ, save_all=False)
        lgp.append(a[0])
        lgp_err.append(a[1])
        nevals.append(a[2])
    brute_mc_res = get_brute_mc_pval(1e4, n_dim=nd, target_obs=targ)
    lgp_mc = brute_mc_res[0]
    lgp_err_mc = brute_mc_res[1]
    with open(pkl_name, 'wb') as pkl:
        pickle.dump((lgp, lgp_err, nevals, lgp_mc, lgp_err_mc), pkl)

mlp, stdlp = np.mean(lgp), np.std(lgp)
# mn_zvals = [norm.isf(10**x) for x in lgp]
mnev, stdnev = np.mean(nevals), np.std(nevals)
merr = np.mean(lgp_err)
p_expected = chi2.sf(targ, df=nd)
lgp_expected = np.log10(p_expected)
errfrommlp = np.sqrt(-np.log(p_expected)/n_live)/np.log(10.0)

print("log10(p) = {:.4f} +/- {:.4f} (avg MN error: {:.4f}); expected: {:.4f} +/- {:.4f}; mc: {:.4f} +/- {:.4f}.".format(mlp, stdlp, merr, lgp_expected, errfrommlp, lgp_mc, lgp_err_mc))
print("Requires {:d} +/- {:d} functions evals.".format(int(mnev),int(stdnev)))

lpvals = np.linspace(mlp-5*stdlp, mlp+5*stdlp, 150)
# zvals = [norm.isf(10**x) for x in lpvals]
bfgauss = norm.pdf(lpvals, loc=mlp, scale=stdlp)
nominal = norm.pdf(lpvals, loc=mlp, scale=errfrommlp)
theory = norm.pdf(lpvals, loc=lgp_expected, scale=errfrommlp)



fig, ax = plt.subplots(figsize=(3.4,2.25))
ax.set_yticks([])

ax.hist(lgp, bins=15, fc='b', ec='none', alpha=0.25, density=True)
ax.plot(lpvals, bfgauss, 'b-', label="Gaussian fit")
# ax.plot(lpvals, nominal, 'r:', label="err formula from mean")
ax.plot(lpvals, theory, 'r--', label="Estimate")
# ax.plot(lpvals, mc, '-', c='orange', label="MC")
# ax.axvline(mlp, c='orange', label="MC")

#plt.title("{:d}-d, Target TS: {:.2f} (~ {:.1f}$\sigma$)".format(nd,targ,significance))
ax.set_xlabel("$\log_{10}(p)$")
ax.set_ylabel("PDF")
ax.set_xlim([mlp-5*stdlp, mlp+5*stdlp])
leg = plt.legend(frameon=False, handlelength=1.9, fontsize=7)
leg.set_title("{:d}-d Gaussian,\ntarget $\chi^2$ = {:.1f}".format(nd,targ), prop={'size': 7})

plt.tight_layout()
plt.savefig("error_check.pdf", backend='pgf')
