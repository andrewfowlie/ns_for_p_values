"""
P-value computation with NS.
"""
import numpy as np

from dynesty import NestedSampler
import pymultinest
import pypolychord
from pypolychord.settings import PolyChordSettings

from .result import Result


def ns_result(n_iter, n_live, calls):
    """
    Results from NS run
    """
    log_x = - float(n_iter) / n_live
    p_value = np.exp(log_x)
    log_x_uncertainty = (-log_x / n_live)**0.5
    p_value_uncertainty = p_value * log_x_uncertainty
    return Result(p_value, p_value_uncertainty, calls)

def log_convergence(threshold, observed):
    """
    Common dumper function
    """
    print("threshold = {}. observed = {}".format(threshold, observed))

def dynesty(test_statistic, transform, n_dim, observed, n_live=100, **kwargs):
    """
    Nested sampling with dynesty
    """
    sampler = NestedSampler(test_statistic, transform, n_dim, nlive=n_live, **kwargs)
    it = 0

    for it, state in enumerate(sampler.sample(dlogz=0.)):

        threshold = state[3]
        calls = state[9]

        if it % 100 == 0:
            log_convergence(threshold, observed)

        if threshold > observed:
            print("threshold reached observed - stopping")
            break

    calls = sampler.results["ncall"].sum()
    n_iter = it + 1
    return ns_result(n_iter, n_live, calls)

def dumper(index, observed):
    """
    Universal function for dumper MN and PC
    """
    def dumper_(*args):
        loglike = args[index][:, -1]
        if loglike.size > 0:
            log_convergence(loglike.min(), observed)
        else:
            print("no live points")
    return dumper_

def stop_at(test_statistic, observed, tuple_=False):
    """
    Implement stopping criteria
    """
    def capped(physical):
        t = test_statistic(physical)

        # force convergence
        if t > observed:
            capped.count += 1
            if capped.count == 1:
                print("capping log-likelihood - beginning to stop")
            t = float(observed)

        if tuple_:
            return t, []
        return t

    capped.count = 0
    return capped

def mn_ev_data(test_statistic, transform, n_dim, max_observed, n_live=100, basename="mn_", resume=False, **kwargs):
    """
    Return the contents of the nested sampling run with MN
    """
    pymultinest.solve(stop_at(test_statistic, max_observed),
                      transform, n_dim, n_live_points=n_live,
                      dump_callback=dumper(3, max_observed),
                      outputfiles_basename=basename,
                      resume=resume,
                      n_iter_before_update=n_live, evidence_tolerance=0., **kwargs)

    ev_data = np.genfromtxt(basename+"ev.dat")

    ts_vals = ev_data[:,-3]
    log_xs = np.arange(0, len(ts_vals), 1.)/n_live
    p_vals = np.exp(-log_xs)
    log_x_uncertainties = np.sqrt(log_xs/n_live)
    p_val_errs = p_vals * log_x_uncertainties

    result = np.array([ts_vals, p_vals, p_val_errs]).T
    return result

def mn(test_statistic, transform, n_dim, observed, n_live=100, basename="mn_", resume=False, **kwargs):
    """
    Nested sampling with MN
    """
    pymultinest.solve(stop_at(test_statistic, observed),
                      transform, n_dim, n_live_points=n_live,
                      dump_callback=dumper(3, observed),
                      outputfiles_basename=basename,
                      resume=resume,
                      n_iter_before_update=n_live, evidence_tolerance=0., **kwargs)

    # get number of iterations
    ev = "{}ev.dat".format(basename)
    with open(ev) as f:
        n_iter = len(f.readlines())

    # get number of calls
    res = "{}resume.dat".format(basename)
    with open(res) as f:
        line = f.readlines()[1]
    calls = int(line.split()[1])

    return ns_result(n_iter, n_live, calls)


def pc(test_statistic, transform, n_dim, observed, n_live=100, file_root="pc_", feedback=0, read_resume=False, **kwargs):
    """
    Nested sampling with PC
    """
    # copy key word arguments to settings object
    settings = PolyChordSettings(n_dim, 0, **kwargs)
    settings.nfail = n_live
    settings.precision_criterion = 0.
    settings.read_resume = read_resume
    settings.file_root = file_root
    settings.nlive = n_live
    settings.feedback = feedback

    output = pypolychord.run_polychord(stop_at(test_statistic, observed, True),
                                       n_dim, 0, settings, transform,
                                       dumper(0, observed))
    n_iter = output.ndead # - n_live  # TODO PC kills final live points?
    calls = output.nlike - (settings.nfail + 1)

    return ns_result(n_iter, n_live, calls)
