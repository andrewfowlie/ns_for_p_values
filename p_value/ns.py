"""
P-value computation with NS.
"""
import numpy as np

from dynesty import NestedSampler
import pymultinest
import pypolychord
from pypolychord.settings import PolyChordSettings

from result import Result


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

def mn_wrap_loglike(test_statistic, observed):
    """
    Implement stopping criteria for MN
    """
    def wrapped(cube, n_dim, n_params, threshold):
        a = np.array([cube[i] for i in range(n_dim)])
        t = float(test_statistic(a))
        wrapped.returned.append(t)
        wrapped.threshold.append(threshold)

        # force convergence
        if t > observed:
            if not wrapped.printed:
                print("capping log-likelihood - beginning to stop")
                wrapped.printed = True
            t = float(observed)
        return t

    wrapped.printed = False
    wrapped.returned = []
    wrapped.threshold = []
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
        wrapped.returned.append(t)
        return (t, [])

    wrapped.returned = []
    return wrapped

def mn(test_statistic, transform, n_dim, observed, n_live=100, basename="mn_", resume=False, ev_data=False, **kwargs):
    """
    Nested sampling with MN
    """
    loglike = mn_wrap_loglike(test_statistic, observed)
    pymultinest.run(loglike,
                    mn_wrap_prior(transform), n_dim, n_live_points=n_live,
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

    if not ev_data:
        return ns_result(n_iter, n_live, calls)

    # get ev data
    ev_data = np.genfromtxt(ev)
    ts = ev_data[:, -3]
    log_x = -np.arange(0, len(ts), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    return ns_result(n_iter, n_live, calls), [ts, log_x, log_x_delta, loglike.returned, loglike.threshold]

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

    return ns_result(n_iter, n_live, calls), [ts, log_x, log_x_delta, loglike.returned]
