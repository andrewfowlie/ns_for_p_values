"""
Fit Higgs to digamma data
=========================
"""

import os
import numpy as np
from numpy.random import default_rng
from scipy.special import binom, erf, xlogy
from scipy.optimize import differential_evolution, minimize

# Digitised data from Fig. 4 in [arXiv:1207.7214]

script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(script_dir+"/atlas_higgs_digamma_data.dat", unpack=True)
edges = np.linspace(100, 160, num=31, endpoint=True)

# Best-fit signals

sigma_from_atlas = 3.9 / (2.0*np.sqrt(2.0*np.log(2.0)))
bkg_bfg = [11., 8.8, 7.6, 6.5, 5.7]
bkg_bounds = 5 * [[0., 75.]]
sig_bfg = [126, 1.0]
sig_bounds = [[110., 150.], [0., 500.]]
bounds = np.array(bkg_bounds + sig_bounds)

def ediff1d(x):
    return x[1:] - x[:-1]

# Functions for modelling background

def bernstein_basis(x, nu, k):
    """
    Berstein polynomials (nu, k) for background model, x in (100, 160) GeV
    """
    # Rescale x to (0, 1)
    xprime = (x - 100.) / 60.
    # Return with a scaling factor ~ O(observed counts)
    return 1.0e4 * binom(k, nu) * xprime**nu * (1. - xprime)**(k - nu)

def fixed_bernstein_basis(x, k):
    jj = np.concatenate([np.arange(nu + 1, k + 1) for nu in range(k)])
    basis = np.array([bernstein_basis(e, jj, k) / k for e in x])

    basis_flat = np.zeros(shape=(len(x), k))
    for place, ii in enumerate(jj):
        basis_flat[:, ii - 1] += basis[:, place]

    return basis_flat

# Compute the basis once and for all at the bin edges
bb = fixed_bernstein_basis(edges, 5)

def background_signal(beta):
    bkg = np.matmul(bb, beta)
    return np.maximum(ediff1d(bkg), 0.)

# Functions for modelling signal

f = 0.5 / np.sqrt(2.0)
edges_over_sigma = edges / sigma_from_atlas

def gaussian_signal(mass, nevents):
    z = edges_over_sigma - mass / sigma_from_atlas
    erf_z = erf(z)
    return f * nevents * ediff1d(erf_z)

# Log-likelihoods

# Get the background expectation value from the best-fitting point (en lieu of theoretical prediction)
beta = np.array([1.10031837e+01, 8.87953673e+00, 7.61866954e+00, 6.49040517e+00, 5.73566527e+00])
expected_bkg = background_signal(beta)

def nan2num(x):
    if np.isinf(x) or np.isnan(x):
        return -1e99
    return x

class loglike(object):
    def __init__(self, data):
        self.data = data
        self.norm = (xlogy(self.data, self.data) - self.data).sum()
    def logpmf(self, x):
        ll = xlogy(self.data, x) - x
        return nan2num(ll.sum() - self.norm)

class loglike_wrapper_spb(loglike):
    def __call__(self, x):
        mass, n_events = x[5], x[6]
        beta = x[:5]
        sig = gaussian_signal(mass, n_events)
        bkg = background_signal(beta)
        spb = sig + bkg
        return -2. * self.logpmf(spb)

class loglike_wrapper_bkg(loglike):
    def __call__(self, x):
        bkg = background_signal(x)
        return -2. * self.logpmf(bkg)

def guess_loc_scale(data, bkg):
    d = data - bkg
    chi = d / bkg**0.5
    ii = chi.argmax()
    loc = edges[ii] + 1.
    scale = d[ii] / gaussian_signal(loc, 1.)[ii] 
    return [loc, scale]
    
def nested_ts(data):

    rng = default_rng()

    polished_0 = minimize(loglike_wrapper_bkg(data), bkg_bfg, method="Nelder-Mead", options={'ftol': 1e-9})
    ts0 = polished_0.fun
    bkg = background_signal(polished_0.x)
    
    signal = list(polished_0.x) + guess_loc_scale(data, bkg)
    no_signal = signal
    no_signal[-1] = 0.
    print(guess_loc_scale(data, bkg))
    draws = rng.uniform(size=(160, 7)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    init = np.array(40 * [signal] + 40 * [no_signal] + draws.tolist())
    de = differential_evolution(loglike_wrapper_spb(data), bounds=bounds, init=init, tol=0.0001)
    polished_1 = minimize(loglike_wrapper_spb(data), de.x, method="Nelder-Mead", options={'ftol': 1e-9})
    ts1 = polished_1.fun
    
    print(polished_0)
    print(polished_1)
    print(ts0, ts1)
    print(ts0 - ts1)
    
    return ts0 - ts1

    
if __name__ == "__main__":
    import time
    b = beta * 1.2
    b[3] *= 0.5
    s = [115., 50.]
    
    t = time.time()
    data = background_signal(b) + gaussian_signal(*s)
    r = nested_ts(data)
    t = time.time() - t
    
    print("time", t)
    print(b, s)
