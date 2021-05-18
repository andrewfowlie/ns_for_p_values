"""
P-value computation with NS via dynesty, PolyChord or MultiNest
===============================================================
"""

import logging
import numpy as np
from scipy.special import logsumexp

from dynesty import NestedSampler
import pymultinest
import pypolychord
from pypolychord.settings import PolyChordSettings

from .result import Result


logging.getLogger().setLevel(logging.DEBUG)


def analyze_mn_output(observed, root="chains/mn_", n_live=100):
    ev_name = "{}ev.dat".format(root)
    ev_data = np.genfromtxt(ev_name)
    n_iter = len(ev_data[:, 0])
    # Only return valid TS values
    cond = ev_data[:, -3] <= observed
    test_statistic = ev_data[cond, -3]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    res_name = "{}resume.dat".format(root)
    with open(res_name) as res_file:
        line = res_file.readlines()[1]
    calls = int(line.split()[1])

    return Result.from_ns(n_iter, n_live, calls), test_statistic, log_x, log_x_delta

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
    return Result.from_ns(n_iter, n_live, calls)

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
            if wrapped.count > 1000 * wrapped.counter:
                logging.debug("threshold = {:.5f} (observed = {:.2f}), calls = {:d} (total = {:d})".format(
                    threshold, observed, int(calls), int(wrapped.count)))
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

def mn(test_statistic, transform, n_dim, observed,
       n_live=100, max_calls=1e10, basename="chains/mn_",
       resume=False, ev_data=False, sampling_efficiency=0.3, **kwargs):
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

    return res, [test_statistic, log_x, log_x_delta, loglike.thresholds, loglike.calls]

def mn_new(test_statistic, transform, n_dim, observed, n_live=100, max_calls=1e8, basename="chains/mn_", resume=False, ev_data=False, sampling_efficiency=0.3, **kwargs):
    """
    Nested sampling with MN
    """

    def safe_transform(cube, ndim, nparams):
       try:
          a = np.array([cube[i] for i in range(n_dim)])
          b = transform(a)
          for i in range(n_dim):
             cube[i] = b[i]
       except Exception as e:
             import sys
             sys.stderr.write('ERROR in prior: %s\n' % e)
             sys.exit(1) 

    def safe_ts(cube, ndim, nparams, threshold):
       try:
          a = np.array([cube[i] for i in range(n_dim)])
          t = float(test_statistic(a))
          if not np.isfinite(t):
             import sys
             sys.stderr.write('WARNING: loglikelihood not finite: %f\n' % (l))
             sys.stderr.write('         for parameters: %s\n' % a)
             sys.stderr.write('         returned very low value instead\n')
             return -1e100
          safe_ts.count += 1
          calls = safe_ts.count - safe_ts.calls[-1]
          if calls > max_calls:
             safe_ts.max_calls_exceeded = True
          # force convergence
          if t > observed or safe_ts.max_calls_exceeded:
             if safe_ts.max_calls_exceeded:
                logging.debug("forcing stop as max calls exceeded")
             else:
                logging.debug("forcing stop as max threshold exceeded")
             t = float(observed)
          if threshold > observed or threshold > safe_ts.threshold:
             safe_ts.threshold = threshold
             safe_ts.thresholds.append(safe_ts.threshold)
             safe_ts.calls.append(safe_ts.count)
             if (safe_ts.count > 1000*safe_ts.counter):
                logging.debug("threshold = {:.5f} (observed = {:.2f}), calls = {:d} (total = {:d})".format(threshold, observed, int(calls), int(safe_ts.count)))
                safe_ts.counter += 1
          return t
       except Exception as e:
          import sys
          sys.stderr.write('ERROR in test statistic evaluation: %s\n' % e)
          sys.exit(1)

    safe_ts.max_calls_exceeded = False
    safe_ts.threshold = -1e30
    safe_ts.thresholds = [-1e30]
    safe_ts.count = 0
    safe_ts.counter = 0
    safe_ts.calls = [0]

    run(safe_ts, safe_transform, n_dim,
          n_live_points=n_live,
          outputfiles_basename=basename,
          dump_callback=dumper(3, observed),
          importance_nested_sampling=False,
          sampling_efficiency=sampling_efficiency,
          evidence_tolerance=0., **kwargs)

    # get number of iterations
    ev_name = "{}ev.dat".format(basename)
    with open(ev_name) as ev_file:
        n_iter = len(ev_file.readlines())

    # get number of calls
    res_name = "{}resume.dat".format(basename)
    with open(res_name) as res_file:
        line = res_file.readlines()[1]
    calls = int(line.split()[1])

    if not ev_data:
        return ns_result(n_iter, n_live, calls)

    # get ev data
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:, -3]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return ns_result(n_iter, n_live, calls), [test_statistic, log_x, log_x_delta, safe_ts.thresholds, safe_ts.calls]

def pc(test_statistic, transform, n_dim, observed, n_live=100, file_root="pc_", feedback=0, resume=False, ev_data=False, base_dir='chains/', **kwargs):
    """
    Nested sampling with PC
    """
    # copy key word arguments to settings object
    settings = PolyChordSettings(n_dim, 0, **kwargs)
    settings.nfail = n_live
    settings.precision_criterion = 0.
    settings.read_resume = resume
    settings.base_dir = base_dir
    settings.file_root = file_root
    settings.nlive = n_live
    settings.logLstop = observed
    settings.do_clustering = False
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
    res_name = "{}/{}.resume".format(base_dir, file_root)

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
        return Result.from_ns(n_iter, n_live, calls)

    # get ev data
    ev_name = "{}/{}_dead.txt".format(base_dir, file_root)
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:, 0]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return Result.from_ns(n_iter, n_live, calls), [test_statistic, log_x, log_x_delta]
