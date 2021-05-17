"""
Make data for chi-squared analytic example
==========================================

The sampling distribution is a set of n_dim draws from a
chi-squared distribution.

The test-statistic is the sum of them, which has a chi^2_{n_dim}
distribution.
"""

import pickle
import numpy as np

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

    rel_error = 0.1
    n_live = 100
    dims = [1, 5, 30]

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

    pkl_name = "pkl/mc.pkl"

    with open(pkl_name, 'wb') as pkl:
        pickle.dump((x, mc, pns), pkl)

    # PolyChord - expensive so save this data

    for d in dims:

        pkl_name = "pkl/pc_dim_{}.pkl".format(d)

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
                p = pc(test_statistic, transform, d, t,
                       n_live=int(n_live), resume=i != 0)
                true_ = analytic_p_value(t, d)

                ns_rel_error = (- np.log(true_.p_value) / n_live)**0.5
                scale = (ns_rel_error / rel_error)**2

               # showing true significance here - could show calculated one
                px.append(true_.significance)
                py.append(p.calls * scale)

            with open(pkl_name, 'wb') as pkl:
                pickle.dump((px, py), pkl)

    for d in dims:

        pkl_name = "pkl/mn_dim_{}.pkl".format(d)

        try:
            with open(pkl_name, 'rb') as pkl:
                px, py = pickle.load(pkl)
        except IOError:
            tmax = chi2.isf(norm.sf(7.), d)
            tmin = chi2.isf(norm.sf(0.), d)

            px = []
            py = []

            # Cannot resume NS run so one long run
            p, ev_data = mn(test_statistic, transform, d, tmax,
                            n_live=int(n_live), basename='chains/mn_d{:d}'.format(d),
                            sampling_efficiency=0.3, ev_data=True, multimodal=False)

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
