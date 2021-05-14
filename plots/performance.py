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

from definitions import *
set_style(lfs=8)

rel_error = 0.1
dims = [1, 5, 30]
ls_list = ['--', ':', '-.']

#plt.style.use("seaborn-colorblind")
#matplotlib.rc('xtick', labelsize=7)
#matplotlib.rc('ytick', labelsize=7)
#matplotlib.rc('legend', fontsize=6)
#matplotlib.rc('axes', labelsize=8, titlesize=9)


fig = plt.figure()
fig.set_size_inches(3.375, 4.25)
gs = gridspec.GridSpec(2, 1, hspace=0., height_ratios=[3., 1.])
ax0 = plt.subplot(gs[0])
ax = [ax0, plt.subplot(gs[1], sharex=ax0)]

#for a in ax:
#    a.grid(lw=0.5, ls=":", zorder=-10, c="grey")

with open("../mc.pkl", 'rb') as pkl:
    x, mc, pns = pickle.load(pkl)
ax[0].plot(x, mc, label="Monte Carlo", c="grey")
ax[0].plot(x, pns, label="Perfect NS", c="k")
ax[1].plot(x, mc / pns, c="k")
ax[1].plot(x, mc / mc, c="grey")


for d,l in zip(dims,ls_list):
    pkl_name = "../pc_dim_{}.pkl".format(d)
    with open(pkl_name, 'rb') as pkl:
        px, py = pickle.load(pkl)

    p = ax[0].plot(px, py, label=r"\textsc{{PolyChord}} ($d = {})$".format(d), ls=l, c='r')
    pr = []
    for t, c in zip(px, py):
        p_value = norm.sf(t)
        mc = 1. / (rel_error**2 * p_value)
        pr.append(mc / c)
    ax[1].plot(px, pr, ls=l, c=p[-1].get_color())

#for a in ax:
#    a.set_prop_cycle(None)

for d,l in zip(dims,ls_list):
    pkl_name = "../mn_dim_{}.pkl".format(d)
    with open(pkl_name, 'rb') as pkl:
        px, py = pickle.load(pkl)

    p = ax[0].plot(px, py, label=r"\textsc{{MultiNest}} ($d = {})$".format(d), ls=l, c='b')
    pr = []
    for t, c in zip(px, py):
        p_value = norm.sf(t)
        mc = 1. / (rel_error**2 * p_value)
        pr.append(mc / c)
    ax[1].plot(px, pr, ls=l, c=p[-1].get_color())


ax[0].axvline(5, lw=1, ls=':', c='k')
ax[1].axvline(5, lw=1, ls=':', c='k')

ax[1].set_ylim(1e-3, 1e9)
ax[0].set_xlim(0, 7)
ax[1].set_yscale('log')
ax[0].set_yscale('log')

ax[0].tick_params(which='both', direction='in', bottom=True, top=False, left=True, right=True)
ax[1].tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)

ax[1].xaxis.set_ticks(range(1,8))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1d$\sigma$'))
major_locator = plt.FixedLocator([10**float(p) for p in np.arange(-6,7,3)])
minor_locator = plt.FixedLocator([10**float(p) for p in range(-5,10)])
ax[1].yaxis.set_major_locator(major_locator)
ax[1].yaxis.set_minor_locator(minor_locator)
#ax[1].yaxis.set_ticks([1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
ax[1].yaxis.set_major_formatter(pow10_formatter)

#ax[0].set_title("Computing $p$-value to 10% uncertainty")
ax[1].set_xlabel("Significance $Z$")
ax[0].set_ylabel("Function calls (proxy for speed)")
ax[1].set_ylabel("Speed-up")

# Show p-value
#pows = range(1, 13, 2)
#p = [10**-x for x in pows]
#z = norm.isf(p)
#labels = ["$p = 10^{{-{}}}$".format(x) for x in pows]
#for zz, l in zip(z, labels):
#    ax[1].axvline(zz, ymin=0, ymax=0.02, lw=1, ls="-", c="grey")
#    ax[1].text(zz - 0.2, 3e-4, l, c="grey", fontsize=5)

secax = ax[0].secondary_xaxis('top', functions=(norm.sf,norm.isf))
secax.tick_params(which='both', direction='in', top=True)
formatter = plt.FuncFormatter(pow10_formatter)
major_locator = plt.FixedLocator([1/10**p for p in [1,3,5,8,11]])
minor_locator = plt.FixedLocator([1/10**p for p in range(1,12)])
secax.xaxis.set_major_locator(major_locator)
secax.xaxis.set_minor_locator(minor_locator)
secax.xaxis.set_major_formatter(formatter)
secax.set_xlabel("$p$-value computed to 10\% uncertainty")

handles, labels = ax[0].get_legend_handles_labels()
ordered = handles
#ordered = handles[2:7]
#ordered.append(handles[0])
#ordered += handles[7:]
#ordered.append(handles[1])
leg = ax[0].legend(handles=ordered, frameon=False, ncol=1, labelspacing=0.55, handlelength=2.2, handletextpad=0.5)#, title="$p$-value computed to 10\% uncertainty") #framealpha=1,
#leg._legend_box.align = "left"

plt.setp(ax[0].get_xticklabels(), visible=False)
plt.setp(ax[1].get_yticklabels(minor=True), visible=False)
plt.tight_layout(pad=0.5)
plt.savefig("performance.pdf", backend='pgf')
