"""
P-value computation with NS via dynesty, PolyChord or MultiNest
===============================================================
"""

import logging
import warnings
import numpy as np
import mpi4py
from scipy.special import logsumexp

from dynesty import NestedSampler
import pymultinest
import pypolychord
from pypolychord.settings import PolyChordSettings

from .result import Result


logging.getLogger().setLevel(logging.DEBUG)


# Note that MN interface won't work with MPI
if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
    warnings.warn("MN analysis of counts and thresholds faulty if running in MPI")


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

def pc_wrap(test_statistic):
    """
    Returns a tuple for PC signature
    """
    def wrapped(physical):
        return (test_statistic(physical), [])

    return wrapped

def mn(test_statistic, transform, n_dim, observed,
       n_live=100, max_calls=1e10, basename="chains/mn_", do_clustering=False,
       resume=False, ev_data=False, sampling_efficiency=0.3, **kwargs):
    """
    Nested sampling with MN.

    Wrappers must be inside scope of function.
    """
    def mn_wrap_prior(cube, ndim, nparams):
        try:
            a = np.array([cube[i] for i in range(n_dim)])
            b = transform(a)
            for i in range(n_dim):
                cube[i] = b[i]
        except Exception as e:
            raise SystemExit('Error in prior: %s' % e)

    def mn_wrap_loglike(cube, ndim, nparams, threshold):
        try:
            a = np.array([cube[i] for i in range(n_dim)])
            t = float(test_statistic(a))

            if not np.isfinite(t):
                logging.warning("log-likelihood not finite: %f\n"
                                "for parameters: %s\n"
                                "returned very low value instead", t, a)
                return -1e100

            mn_wrap_loglike.count += 1
            calls = mn_wrap_loglike.count - mn_wrap_loglike.calls[-1]

            if calls > max_calls:
                mn_wrap_loglike.max_calls_exceeded = True

            # force convergence
            if t > observed or mn_wrap_loglike.max_calls_exceeded:
                if mn_wrap_loglike.max_calls_exceeded:
                    logging.debug("forcing stop as max calls exceeded")
                else:
                    logging.debug("forcing stop as max threshold exceeded")
                t = float(observed)

            if threshold > observed or threshold > mn_wrap_loglike.threshold:
                mn_wrap_loglike.threshold = threshold
                mn_wrap_loglike.thresholds.append(mn_wrap_loglike.threshold)
                mn_wrap_loglike.calls.append(mn_wrap_loglike.count)
                if mn_wrap_loglike.count > 1000*mn_wrap_loglike.counter:
                    logging.debug("threshold = {:.5f} (observed = {:.2f}), calls = {:d} (total = {:d})".format(
                        threshold, observed, int(calls), int(mn_wrap_loglike.count)))
                    mn_wrap_loglike.counter += 1

            return t

        except Exception as e:
            raise SystemExit('Error in test statistic evaluation: %s' % e)

    mn_wrap_loglike.max_calls_exceeded = False
    mn_wrap_loglike.threshold = -1e30
    mn_wrap_loglike.thresholds = [-1e30]
    mn_wrap_loglike.count = 0
    mn_wrap_loglike.counter = 0
    mn_wrap_loglike.calls = [0]

    pymultinest.run(mn_wrap_loglike, mn_wrap_prior, n_dim,
                    n_live_points=n_live,
                    outputfiles_basename=basename,
                    dump_callback=dumper(3, observed),
                    importance_nested_sampling=False,
                    sampling_efficiency=sampling_efficiency,
                    multimodal=do_clustering,
                    evidence_tolerance=0.,
                    resume=resume, **kwargs)

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
        return Result.from_ns(n_iter, n_live, calls)

    # get ev data
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:, -3]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return Result.from_ns(n_iter, n_live, calls), [test_statistic, log_x, log_x_delta, mn_wrap_loglike.thresholds, mn_wrap_loglike.calls]

def get_line(file, label):
    with open(file) as res_file:
        for line in res_file:
            if line.strip() == label:
                next_line = res_file.readline()
                return np.array([float(e) for e in next_line.split()])
        else:
            raise RuntimeError("Did not find the line '{}'".format(label))

def analyse_pc_output(root="pc_", base_dir="chains/", n_live=100):
    """
    Analyse output of PC runs
    """
    label1 = "=== local volume -- log(<X_p>) ==="
    label2 = "=== Number of likelihood calls ==="

    log_xp = get_line("{}/{}.resume".format(base_dir, file_root), label1)
    log_x = logsumexp(log_xp)
    n_iter = -log_x * n_live

    # Get evidence data
    ev_name = "{}/{}_dead.txt".format(base_dir, file_root)
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:,0]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    # Get number of PC calls
    calls = int(get_line("chains/"+root+".resume", label2)[0])

    return Result.from_ns(n_iter, n_live, calls), test_statistic, log_x, log_x_delta

def pc(test_statistic, transform, n_dim, observed,
       n_live=100, base_dir="chains/", file_root="pc_", do_clustering=False,
       resume=False, ev_data=False, feedback=0, **kwargs):
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
    settings.do_clustering = do_clustering
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
