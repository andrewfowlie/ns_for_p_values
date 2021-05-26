"""
=============================================================
Calibration of the TS for Higgs resonance search like problem
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from definitions import set_style, log10_special_formatter_every_n

def log10p_to_significance(lgp):
    return norm.isf(np.minimum(10**lgp,0.5))-0.055 # Include a fudge factor for plotting artefact

def significance_to_log10p(z):
    return np.log10(norm.sf(z))

set_style()

ts_vals, mc_lgpval, mc_lgpval_err_lo, mc_lgpval_err_hi, pch_lgpvals, pch_lgpvals_err,\
mn_lgpvals, mn_lgpvals_err = np.genfromtxt('data/resonance_search_pvals.txt', unpack=True)

fig, ax = plt.subplots(figsize=(3.375,2.1))

# Plot the Monte Carlo results with Clopper-Pearson error bands
ax.plot(ts_vals, mc_lgpval, c='grey', ls='-', label='Monte Carlo')
ax.fill_between(ts_vals, mc_lgpval_err_lo, mc_lgpval_err_hi, fc='grey', alpha=0.25)

# Plot the Nested Sampling results with error bands
ax.plot(ts_vals, pch_lgpvals, c='red', label=r'\textsc{PolyChord}')
ax.fill_between(ts_vals, pch_lgpvals-pch_lgpvals_err, pch_lgpvals+pch_lgpvals_err, fc='red', alpha=0.25)
ax.plot(ts_vals, mn_lgpvals, c='blue', label=r'\textsc{MultiNest}')
ax.fill_between(ts_vals, mn_lgpvals-mn_lgpvals_err, mn_lgpvals+mn_lgpvals_err, fc='blue', alpha=0.25)

# The 5sigma line
p_5sigma = norm.sf(5)
ax.axhline(y=np.log10(p_5sigma), c='k', ls=':', lw=1)

ax.legend(loc=1, frameon=False, handlelength=2.7)
ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=False)
ax.set_xlabel(r'Test statistic $\lambda$')
ax.set_ylabel(r'$P$-value')
ax.set_xlim([0,50])
ax.set_ylim([-10.5,0])

formatter = plt.FuncFormatter(log10_special_formatter_every_n)
major_locator = plt.FixedLocator(range(-10,1))
minor_locator = plt.FixedLocator([i+j for i in np.log10(range(2,11)) for j in range(-11,1)])
ax.yaxis.set_major_locator(major_locator)
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.set_major_formatter(formatter)
secax = ax.secondary_yaxis('right', functions=(log10p_to_significance, significance_to_log10p))
secax.tick_params(which='both', direction='in', right=True)
formatter = plt.FormatStrFormatter('%.1d$\sigma$')
major_locator = plt.FixedLocator(range(1,8))
secax.yaxis.set_major_locator(major_locator)
secax.yaxis.set_major_formatter(formatter)
secax.set_ylabel("Significance $Z$", rotation=270, labelpad=15)

plt.tight_layout(pad=0.2)
plt.savefig('higgs_nested_vs_mc.pdf', backend='pgf')
