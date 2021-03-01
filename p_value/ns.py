"""
P-value computation with NS.
"""
import numpy as np

from dynesty import NestedSampler
import pymultinest

from .result import Result

def dynesty(test_statistic, transform, n_dim, observed, n_live=100, **kwargs):
    """
    Nested sampling with dynesty
    """
    sampler = NestedSampler(test_statistic, transform, n_dim, n_live=n_live, **kwargs)

    for it, state in enumerate(sampler.sample(dlogz=1e-10)):

        threshold = state[3]
        log_x = state[4]
        calls = state[9]

        if it % 100 == 0:
            print("it = ", it,
                  ". threshold = ", threshold,
                  ". observed_test_statistic = ", observed,
                  ". log_x = ", log_x,
                  ". calls = ", calls)

        if threshold > observed:
            break

    p_value = np.exp(log_x)
    log_x_uncertainty = (-log_x / n_live)**0.5
    p_value_uncertainty = p_value * log_x_uncertainty
    calls = sampler.results["ncall"].sum()
    return Result(p_value, p_value_uncertainty, calls)
    
    
def mn(test_statistic, transform, n_dim, observed, n_live=100, **kwargs):
    """
    Nested sampling with MN
    """
    def transform_(cube, ndim, nparams):
        np_cube = np.array([cube[i] for i in range(ndim)])
        transformed = transform(np_cube)
        for i in range(ndim):
            cube[i] = transformed[i]
         
    def test_statistic_(physical, ndim, nparams, lnew):
       
        # force convergence by making all new live poins have same massive likelihood
        force_loglike = 1e100
        if lnew > observed: 
            print("Forcing convergence")
            return force_loglike
            
        np_physical = np.array([physical[i] for i in range(ndim)])
        return test_statistic(np_physical)

    def dumper(nSamples, nlive, nPar, physLive, posterior, paramConstr, maxLogLike, logZ, logZerr, nullcontext):
        lnew = physLive[:, -1].min()
        print("threshold = ", lnew)

    pymultinest.run(test_statistic_, transform_, n_dim, n_live_points=n_live, verbose=True, dump_callback=dumper, resume=False, n_iter_before_update=n_live, evidence_tolerance=0., **kwargs)
    
    # get number of iterations
    ev = "chains/1-ev.dat"
    with open(ev) as f:
        n_iter = len(f.readlines()) 
    n_iter -= n_live  # bogus calls to force convergence
    
    # get number of calls
    res = "chains/1-resume.dat"
    with open(res) as f:
        line = f.readlines()[1]
    calls = int(line.split()[1]) 
    calls -= n_live  # bogus calls to force convergence
    
    log_x = - float(n_iter) / n_live
    p_value = np.exp(log_x)
    log_x_uncertainty = (-log_x / n_live)**0.5
    p_value_uncertainty = p_value * log_x_uncertainty
    return Result(p_value, p_value_uncertainty, calls)
