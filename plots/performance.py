"""
Plot significance against calls for MN and PC
=============================================

Requires data saved as pikcles.
"""

import numpy as np

import pickle

from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

from definitions import set_style, pow10_formatter

set_style(lfs=8)

rel_error = 0.1
dims = [1, 5, 30]
ls_list = ['--', ':', '-.']

fig = plt.figure()
fig.set_size_inches(3.375, 4)
gs = gridspec.GridSpec(2, 1, hspace=0., height_ratios=[3., 1.])
ax0 = plt.subplot(gs[0])
ax = [ax0, plt.subplot(gs[1], sharex=ax0)]

with open("../examples/pkl/mc.pkl", 'rb') as pkl:
    x, mc, pns = pickle.load(pkl)

ax[0].plot(x, mc, label="Monte Carlo", c="grey", zorder=10)
ax[0].plot(x, pns, label="Perfect NS", c="k")
ax[1].plot(x, mc / pns, c="k")
ax[1].plot(x, mc / mc, c="grey")


for d, l in zip(dims, ls_list):
    try:
        txt_name = "./data/pc_dim_{}.txt".format(d)
        px, py = np.loadtxt(txt_name)

        p = ax[0].plot(px, py, label=r"\textsc{{PolyChord}} ($d = {})$".format(d), ls=l, c='r', zorder=5)
        pr = []
        for t, c in zip(px, py):
            p_value = norm.sf(t)
            mc = 1. / (rel_error**2 * p_value)
            pr.append(mc / c)
        ax[1].plot(px, pr, ls=l, c=p[-1].get_color())
    except IOError:
        continue

for d, l in zip(dims, ls_list):
    try:
        txt_name = "./data/mn_dim_{}.txt".format(d)
        px, py = np.loadtxt(txt_name)

        p = ax[0].plot(px, py, label=r"\textsc{{MultiNest}} ($d = {})$".format(d), ls=l, c='b')
        pr = []
        for t, c in zip(px, py):
            p_value = norm.sf(t)
            mc = 1. / (rel_error**2 * p_value)
            pr.append(mc / c)
        ax[1].plot(px, pr, ls=l, c=p[-1].get_color())
    except IOError:
        continue

ax[0].axvline(5, lw=1, ls=':', c='k')
ax[1].axvline(5, lw=1, ls=':', c='k')

ax[0].set_ylim(1e1, 1e14)
ax[1].set_ylim(1e-3, 1e9)
ax[0].set_xlim(0, 7)
ax[1].set_yscale('log')
ax[0].set_yscale('log')

ax[0].tick_params(which='both', direction='in', bottom=True, top=False, left=True, right=True)
ax[1].tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)

ax[1].xaxis.set_ticks(range(1, 8))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1d$\sigma$'))
major_locator = plt.FixedLocator([10**float(p) for p in np.arange(-6, 7, 3)])
minor_locator = plt.FixedLocator([10**float(p) for p in range(-5, 10)])
ax[1].yaxis.set_major_locator(major_locator)
ax[1].yaxis.set_minor_locator(minor_locator)
ax[1].yaxis.set_major_formatter(pow10_formatter)
ax[0].tick_params(axis='y', which='minor', left=True, direction='in')
ax[0].set_yticks([10**p for p in range(1, 15, 1)], minor=True)
ax[0].set_yticks([10**p for p in range(3, 15, 2)])

ax[1].set_xlabel("Significance $Z$")
ax[0].set_ylabel("TS evaluations (proxy for speed)")
ax[1].set_ylabel("Speed-up")

secax = ax[0].secondary_xaxis('top', functions=(norm.sf, norm.isf))
secax.tick_params(which='both', direction='in', top=True, pad=2.25, labelsize=9)
formatter = plt.FuncFormatter(pow10_formatter)
major_locator = plt.FixedLocator([1/10**p for p in [1, 4, 7, 10]])
minor_locator = plt.FixedLocator([1/10**p for p in range(1, 12)])
secax.xaxis.set_major_locator(major_locator)
secax.xaxis.set_minor_locator(minor_locator)
secax.xaxis.set_major_formatter(formatter)
secax.set_xlabel(r"\textit{P-}value computed to 10\% uncertainty", labelpad=6.)

handles, labels = ax[0].get_legend_handles_labels()
ordered = handles
leg = ax[0].legend(handles=ordered, frameon=False, ncol=1, labelspacing=0.55, handlelength=1.9, handletextpad=0.5)

for a in ax:
    a.set_axisbelow(False)
plt.setp(ax[0].get_xticklabels(), visible=False)
plt.setp(ax[0].get_yticklabels(minor=True), visible=False)
plt.setp(ax[1].get_yticklabels(minor=True), visible=False)
plt.tight_layout(pad=0.25)
plt.savefig("performance.pdf", backend='pgf')
