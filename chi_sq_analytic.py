import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from scipy.stats import chi2, norm
from scipy.special import chdtri

from p_value import mn, pc, Result


def analytic_p_value(observed_, n_dim):
    p = chi2.sf(observed_, n_dim)
    return Result(p, 0, None)

def transform(cube):
    return chdtri(1, cube)

def test_statistic(observed_):
    return observed_.sum()


if __name__ == "__main__":

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

    rel_error = 0.1
    n_live = 100
    dims = [1, 2, 5, 10, 30]

    # MC and perfect NS

    d = 1  # independent of d in this setting
    tmax = chi2.isf(norm.sf(7.), d)
    tmin = chi2.isf(norm.sf(0.), d)
    x = []
    mc = []
    pns = []

    for t in np.geomspace(tmin, tmax, 500):
        r = analytic_p_value(t, d)
        x.append(r.significance)

        mc.append(1. / (rel_error**2 * r.p_value))

        n_live = - np.log(r.p_value) / rel_error**2
        calls = -n_live * np.log(r.p_value)
        pns.append(calls)

    mc = np.array(mc)
    pns = np.array(pns)
    ax[0].plot(x, mc, label="Monte Carlo", c="Crimson")
    ax[0].plot(x, pns, label="Perfect NS", c="k")
    ax[1].plot(x, mc / pns, label="Perfect NS", c="k")
    ax[1].plot(x, mc / mc, label="Monte Carlo", c="Crimson")

    # PolyChord - expensive so save this data

    for d in dims:

        pkl_name = "pc_dim_{}.pkl".format(d)

        try:
            with open(pkl_name, 'rb') as pkl:
                px, py = pickle.load(pkl)
        except IOError:
            tmax = chi2.isf(norm.sf(7.), d)
            tmin = chi2.isf(norm.sf(0.), d)

            px = []
            py = []

            for i, t in enumerate(np.geomspace(tmin, tmax, 20)):

                # Strategy is resume NS run, pushing threshold a bit further
                p = pc(test_statistic, transform, d, t, n_live=int(n_live), resume=i != 0)
                true_ = analytic_p_value(t, d)

                ns_rel_error = (- np.log(true_.p_value) / n_live)**0.5
                scale = (ns_rel_error / rel_error)**2

                # showing true significance here - could show calculated one
                px.append(true_.significance)
                py.append(p.calls * scale)

            with open(pkl_name, 'wb') as pkl:
                pickle.dump((px, py), pkl)

        p = ax[0].plot(px, py, label="PolyChord. $d = {}$".format(d), ls="--")

        pr = []
        for t, c in zip(px, py):
            p_value = norm.sf(t)
            mc = 1. / (rel_error**2 * p_value)
            pr.append(mc / c)

        ax[1].plot(px, pr, label="PolyChord. $d = {}$".format(d), ls="--", c=p[-1].get_color())

    # MultiNest - expensive so save this data

    # Reset colors so same as PolyChord
    for a in ax:
        a.set_prop_cycle(None)

    for d in dims:

        pkl_name = "mn_dim_{}.pkl".format(d)

        try:
            with open(pkl_name, 'rb') as pkl:
                px, py = pickle.load(pkl)
        except IOError:
            tmax = chi2.isf(norm.sf(7.), d)
            tmin = chi2.isf(norm.sf(0.), d)

            px = []
            py = []

            # Cannot resume NS run so one long run
            p, ev_data = mn(test_statistic, transform, d, tmax, n_live=int(n_live), max_calls=1e3/0.3, sampling_efficiency=0.3, ev_data=True)

            # extract number of calls
            thresholds = ev_data[-2]
            calls = ev_data[-1]

            for t, c in zip(thresholds, calls):

                if t < tmin or t > tmax:
                    continue

                true_ = analytic_p_value(t, d)

                ns_rel_error = (- np.log(true_.p_value) / n_live)**0.5
                scale = (ns_rel_error / rel_error)**2

                # showing true significance here - could show calculated one
                px.append(true_.significance)
                py.append(c * scale)

            with open(pkl_name, 'wb') as pkl:
                pickle.dump((px, py), pkl)

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
