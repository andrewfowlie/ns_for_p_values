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

def mn_wrap_loglike(test_statistic, observed, max_calls):
    """
    Implement stopping criteria for MN
    """
    def wrapped(cube, n_dim, n_params, threshold):
        a = np.array([cube[i] for i in range(n_dim)])
        t = float(test_statistic(a))
        wrapped.count += 1

        calls = wrapped.count - wrapped.calls[-1]
        if calls > max_calls:
            wrapped.max_calls_exceeded = True

        # force convergence
        if t > observed or wrapped.max_calls_exceeded:
            if wrapped.max_calls_exceeded:
                print("forcing stop as max calls exceeded")
            else:
                print("forcing stop as max threshold exceeded")
            t = float(observed)

        if threshold > observed or threshold > wrapped.threshold:
            wrapped.threshold = threshold
            wrapped.thresholds.append(wrapped.threshold)
            wrapped.calls.append(wrapped.count)
            print("threshold = {:.2f}. observed = {:.2f}. calls = {:.2f}. total = {:.2f}".format(threshold, observed, calls, wrapped.count), end="\r")

        return t

    wrapped.max_calls_exceeded = False
    wrapped.threshold = -1e30
    wrapped.thresholds = [-1e30]
    wrapped.count = 0
    wrapped.calls = [0]
    return wrapped

def mn_wrap_prior(transform):
    """
    Safely wrap MN prior
    """
    def wrapped(cube, n_dim, n_params):
        a = np.array([cube[i] for i in range(n_params)])
        b = transform(a)
        for i in range(n_params):
            cube[i] = b[i]

    return wrapped

def pc_wrap(test_statistic):
    """
    Returns a tuple for PC signature
    """
    def wrapped(physical):
        t = test_statistic(physical)
        return (t, [])

    return wrapped

def mn(test_statistic, transform, n_dim, observed, n_live=100, max_calls=1e8, basename="chains/mn_", resume=False, ev_data=False, **kwargs):
    """
    Nested sampling with MN
    """
    loglike = mn_wrap_loglike(test_statistic, observed, max_calls)
    pymultinest.run(loglike,
                    mn_wrap_prior(transform), n_dim, n_live_points=n_live,
                    dump_callback=dumper(3, observed),
                    outputfiles_basename=basename,
                    resume=resume,
                    evidence_tolerance=0., **kwargs)

    # get number of iterations
    ev = "{}ev.dat".format(basename)
    with open(ev) as f:
        n_iter = len(f.readlines())

    # get number of calls
    res = "{}resume.dat".format(basename)
    with open(res) as f:
        line = f.readlines()[1]
    calls = int(line.split()[1])

    if not ev_data:
        return ns_result(n_iter, n_live, calls)

    # get ev data
    ev_data = np.genfromtxt(ev)
    ts = ev_data[:, -3]
    log_x = -np.arange(0, len(ts), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return ns_result(n_iter, n_live, calls), [ts, log_x, log_x_delta, loglike.thresholds, loglike.calls]

def pc(test_statistic, transform, n_dim, observed, n_live=100, file_root="pc_", feedback=0, resume=False, ev_data=False, **kwargs):
    """
    Nested sampling with PC
    """
    # copy key word arguments to settings object
    settings = PolyChordSettings(n_dim, 0, **kwargs)
    settings.nfail = n_live
    settings.precision_criterion = 0.
    settings.read_resume = resume
    settings.file_root = file_root
    settings.nlive = n_live
    settings.feedback = feedback
    settings.logLstop = observed

    loglike = pc_wrap(test_statistic)
    output = pypolychord.run_polychord(loglike,
                                       n_dim, 0, settings, transform,
                                       dumper(0, observed))
    n_iter = output.ndead - n_live  # PC kills final live points so subtract that
    calls = output.nlike

    if not ev_data:
        return ns_result(n_iter, n_live, calls)

    # get ev data
    ev = "chains/{}_dead.txt".format(file_root)
    ev_data = np.genfromtxt(ev)
    ts = ev_data[:, 0]
    log_x = -np.arange(0, len(ts), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return ns_result(n_iter, n_live, calls), [ts, log_x, log_x_delta]
