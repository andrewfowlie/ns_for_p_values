"""
Plot significance against calls.
"""

import pickle
import numpy as np

from scipy.stats import norm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

rel_error = 0.1
dims = [1, 2]

# Plot significance against calls

plt.style.use("seaborn-colorblind")

matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)
matplotlib.rc('legend', fontsize=6)
matplotlib.rc('axes', labelsize=8, titlesize=9)


fig = plt.figure()
gs = gridspec.GridSpec(2, 1, hspace=0., height_ratios=[3., 1.])
ax0 = plt.subplot(gs[0])
ax = [ax0, plt.subplot(gs[1], sharex=ax0)]

for a in ax:
    a.grid(lw=0.5, ls=":", zorder=-10, c="grey")

with open("../mc.pkl", 'rb') as pkl:
    x, mc, pns = pickle.load(pkl)
ax[0].plot(x, mc, label="Monte Carlo", c="Crimson")
ax[0].plot(x, pns, label="Perfect NS", c="k")
ax[1].plot(x, mc / pns, label="Perfect NS", c="k")
ax[1].plot(x, mc / mc, label="Monte Carlo", c="Crimson")


for d in dims:
    pkl_name = "../pc_dim_{}.pkl".format(d)
    with open(pkl_name, 'rb') as pkl:
        px, py = pickle.load(pkl)

    p = ax[0].plot(px, py, label="PolyChord. $d = {}$".format(d), ls="--")
    pr = []
    for t, c in zip(px, py):
        p_value = norm.sf(t)
        mc = 1. / (rel_error**2 * p_value)
        pr.append(mc / c)
    ax[1].plot(px, pr, label="PolyChord. $d = {}$".format(d), ls="--", c=p[-1].get_color())

for a in ax:
    a.set_prop_cycle(None)

for d in dims:
    pkl_name = "../mn_dim_{}.pkl".format(d)
    with open(pkl_name, 'rb') as pkl:
        px, py = pickle.load(pkl)

    p = ax[0].plot(px, py, label="MultiNest. $d = {}$".format(d), ls=":")
    pr = []
    for t, c in zip(px, py):
        p_value = norm.sf(t)
        mc = 1. / (rel_error**2 * p_value)
        pr.append(mc / c)
    ax[1].plot(px, pr, label="MultiNest. $d = {}$".format(d), ls=":", c=p[-1].get_color())

handles, labels = ax[0].get_legend_handles_labels()
ordered = handles[2:7]
ordered.append(handles[0])
ordered += handles[7:]
ordered.append(handles[1])
leg = ax[0].legend(handles=ordered, framealpha=1., ncol=2, handleheight=1.7)

ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1d$\sigma$'))
ax[0].set_title("Computing $p$-value to 10% uncertainty")
ax[1].set_xlabel("Significance, $Z$")
ax[0].set_ylabel("Function calls (proxy for speed)")
ax[1].set_ylabel("Speed-up")
ax[1].set_yscale('log')
ax[0].set_yscale('log')

# Alter tick locations
ax[1].yaxis.set_ticks([1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
ax[1].set_ylim(1e-4, 1e9)

# Show p-value
pows = range(1, 13, 2)
p = [10**-x for x in pows]
z = norm.isf(p)
labels = ["$p = 10^{{-{}}}$".format(x) for x in pows]

for zz, l in zip(z, labels):
    ax[1].axvline(zz, ymin=0, ymax=0.02, lw=1, ls="-", c="grey")
    ax[1].text(zz - 0.2, 3e-4, l, c="grey", fontsize=5)

plt.setp(ax[0].get_xticklabels(), visible=False)
fig.set_size_inches(4, 5)
plt.tight_layout()
plt.savefig("performance.pdf")
