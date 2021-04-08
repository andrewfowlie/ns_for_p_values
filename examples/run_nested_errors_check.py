import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, chi2

from nested_errors import get_mn_pval_single, get_brute_mc_pval


nd = 4
targ = 9.0
n_live = 100
obs = chi2.isf(chi2.sf(targ, df=1), df=nd)

logp, logp_err = [], []
for i in tqdm(range(200)):
    a = get_mn_pval_single(i, file_root="./temp/mn", n_dim=nd, target_obs=targ, save_all=False)
    logp.append(a[0])
    logp_err.append(a[1])

mlp = np.mean(logp)
stdlp = np.std(logp)
merr = np.mean(logp_err)
expected = np.log10(chi2.sf(targ, df=nd))
errfrommlp = (-expected/n_live)**0.5
brute_mc_res = get_brute_mc_pval(1e4, n_dim=nd)
mc_lp = brute_mc_res[0]
mc_lperr = brute_mc_res[1]

print("log10(p) = {:.4f} +/- {:.4f} (avg MN error: {:.4f}); expected: {:.4f} +/- {:.4f}; mc: {:.4f} +/- {:.4f}".format(mlp, stdlp, merr, expected, errfrommlp, mc_lp, mc_lperr))

lpvals = np.linspace(mlp-5*stdlp, mlp+5*stdlp, 150)
bfgauss = norm.pdf(lpvals, loc=mlp, scale=stdlp)
nominal = norm.pdf(lpvals, loc=mlp, scale=errfrommlp)
theory = norm.pdf(lpvals, loc=expected, scale=errfrommlp)
# mc = norm.pdf(lpvals, loc=mc_lp, scale=mc_lperr)

fig, ax = plt.subplots(figsize=(6,3.5))

ax.hist(logp, density=True)
ax.plot(lpvals, bfgauss, 'r-', label="best fit")
ax.plot(lpvals, nominal, 'r:', label="err formula from mean")
ax.plot(lpvals, theory, 'r--', label="analytical")
# ax.plot(lpvals, mc, '-', c='orange', label="MC")
ax.axvline(mlp, c='orange', label="MC")

plt.title("{:d}-d, Target TS: {:.2f} (~ {:.1f}$\sigma$)".format(nd,targ,norm.isf(chi2.sf(targ, df=nd))))
plt.xlabel("$\log(p\mathrm{-value})$")
plt.ylabel("Relative prob. dens.")
plt.legend(frameon=False)

plt.savefig("error_check.pdf")
plt.show()
