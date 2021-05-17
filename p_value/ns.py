"""
P-value computation with NS.
"""

import logging
import numpy as np
from scipy.special import logsumexp

from dynesty import NestedSampler
import pymultinest
from pymultinest.run import run
import pypolychord
from pypolychord.settings import PolyChordSettings

from .result import Result


logging.getLogger().setLevel(logging.DEBUG)


def ns_result(n_iter, n_live, calls):
    """
    Results from NS run
    """
    log_x = - float(n_iter) / n_live
    p_value = np.exp(log_x)
    log_x_uncertainty = (-log_x / n_live)**0.5
    p_value_uncertainty = p_value * log_x_uncertainty
    return Result(p_value, p_value_uncertainty, calls)

def analyze_mn_output(observed, root="chains/mn_", n_live=100):
    ev_name = "{}ev.dat".format(root)
    ev_data = np.genfromtxt(ev_name)
    n_iter = len(ev_data[:,0])
    # Only return valid TS values
    cond = ev_data[:,-3] <= observed
    test_statistic = ev_data[cond,-3]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    res_name = "{}resume.dat".format(root)
    with open(res_name) as res_file:
        line = res_file.readlines()[1]
    calls = int(line.split()[1])

    return ns_result(n_iter, n_live, calls), test_statistic, log_x, log_x_delta

def log_convergence(threshold, observed):
    """
    Common dumper function
    """
    logging.info("threshold = %s. observed = %s", threshold, observed)

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
            logging.info("threshold reached observed - stopping")
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
            logging.info("no live points")
    return dumper_

def mn_wrap_loglike(test_statistic, observed, max_calls):
    """
    Implement stopping criteria for MN
    """
    def wrapped(cube, n_dim, n_params, threshold):
        cube_arr = np.array([cube[i] for i in range(n_dim)])
        t = float(test_statistic(cube_arr))
        wrapped.count += 1

        calls = wrapped.count - wrapped.calls[-1]
        if calls > max_calls:
            wrapped.max_calls_exceeded = True

        # force convergence
        if t > observed or wrapped.max_calls_exceeded:
            if wrapped.max_calls_exceeded:
                logging.debug("forcing stop as max calls exceeded")
            else:
                logging.debug("forcing stop as max threshold exceeded")
            t = float(observed)

        if threshold > observed or threshold > wrapped.threshold:
            wrapped.threshold = threshold
            wrapped.thresholds.append(wrapped.threshold)
            wrapped.calls.append(wrapped.count)
            if (wrapped.count > 1000*wrapped.counter):
                logging.debug("threshold = {:.5f} (observed = {:.2f}), calls = {:d} (total = {:d})".format(threshold, observed, int(calls), int(wrapped.count)))
                wrapped.counter += 1
        return t

    wrapped.max_calls_exceeded = False
    wrapped.threshold = -1e30
    wrapped.thresholds = [-1e30]
    wrapped.count = 0
    wrapped.counter = 0
    wrapped.calls = [0]
    return wrapped

def mn_wrap_prior(transform):
    """
    Safely wrap MN prior
    """
    def wrapped(cube, n_dim, n_params):
        cube_arr = np.array([cube[i] for i in range(n_params)])
        phys_arr = transform(cube_arr)
        for i in range(n_params):
            cube[i] = phys_arr[i]

    return wrapped

def pc_wrap(test_statistic):
    """
    Returns a tuple for PC signature
    """
    def wrapped(physical):
        return (test_statistic(physical), [])

    return wrapped

def mn(test_statistic, transform, n_dim, observed, n_live=100, max_calls=1e8, basename="chains/mn_", resume=False, ev_data=False, sampling_efficiency=0.3, **kwargs):
    """
    Nested sampling with MN
    """
    loglike = mn_wrap_loglike(test_statistic, observed, max_calls)
    pymultinest.run(loglike,
                    mn_wrap_prior(transform), n_dim, n_live_points=n_live,
                    dump_callback=dumper(3, observed),
                    outputfiles_basename=basename,
                    resume=resume,
                    importance_nested_sampling=False,
                    sampling_efficiency=sampling_efficiency,
                    evidence_tolerance=0., **kwargs)

    res, test_statistic, log_x, log_x_delta = analyze_mn_output(observed, root=basename, n_live=n_live)

    if not ev_data:
        return res
    else:
        return res, [test_statistic, log_x, log_x_delta, loglike.thresholds, loglike.calls]

def pc(test_statistic, transform, n_dim, observed, n_live=100, file_root="pc_", feedback=0, resume=False, ev_data=False, **kwargs):
    """
    Nested sampling with PC
    """
    # copy key word arguments to settings object
    settings = PolyChordSettings(n_dim, 0, **kwargs)
    settings.nfail = n_live
    settings.precision_criterion = 0.
    settings.read_resume = resume
    settings.nlive = n_live
    settings.logLstop = observed
    settings.do_clustering = False
    settings.file_root = file_root
    settings.feedback = feedback

    loglike = pc_wrap(test_statistic)
    output = pypolychord.run_polychord(loglike,
                                       n_dim, 0, settings, transform,
                                       dumper(0, observed))

    # get number of calls directly
    calls = output.nlike

    # get log X from resume file

    label = "=== local volume -- log(<X_p>) ==="
    log_xp = None
    res_name = "chains/{}.resume".format(file_root)

    with open(res_name) as res_file:
        for line in res_file:
            if line.strip() == label:
                next_line = res_file.readline()
                log_xp = np.array([float(e) for e in next_line.split()])
                break
        else:
            raise RuntimeError("didn't find {}".format(label))

    log_x = logsumexp(log_xp)
    n_iter = -log_x * n_live

    if not ev_data:
        return ns_result(n_iter, n_live, calls)

    # get ev data
    ev_name = "chains/{}_dead.txt".format(file_root)
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:, 0]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return ns_result(n_iter, n_live, calls), [test_statistic, log_x, log_x_delta]
