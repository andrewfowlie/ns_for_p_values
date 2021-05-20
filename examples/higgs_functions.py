"""
Fit Higgs to digamma data
=========================
"""

import os
import numpy as np
from numpy.random import default_rng
from scipy.special import binom, erf, gammaln, xlogy
from scipy.stats import poisson
from scipy.optimize import differential_evolution, minimize

# Digitised data from Fig. 4 in [arXiv:1207.7214]

script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(script_dir+"/atlas_higgs_digamma_data.dat", unpack=True)
bins = np.linspace(100, 160, num=30, endpoint=False)
edges = bins.tolist() + [bins[-1] + 2.]

def logpmf(k, mu, gammaln_sum):
    ll = xlogy(k, mu) -  mu
    return ll.sum() - gammaln_sum

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

def bernstein_integral(beta):
    return (beta * bb).sum(axis=-1)

def background_signal(beta):
    bkg = bernstein_integral(beta)
    return np.maximum(np.ediff1d(bkg), 0.)

# Functions for modelling Higgs signal as a Gaussian peak

def gaussian_signal_density(x, mu, sigma, n_events):
    n = n_events/(np.sqrt(2.0*np.pi)*sigma)
    z = (x-mu)/sigma
    return n*np.exp(-0.5*z*z)

def gaussian_signal_integral(x, mu, sigma):
    z = (x-mu)/(np.sqrt(2.0)*sigma)
    return 0.5 *  erf(z)

def gaussian_signal(edges, mu, sigma, nevents):
    sig = [gaussian_signal_integral(e, mu, sigma) for e in edges]
    return nevents * np.maximum(np.ediff1d(sig), 0.)

# Log-likelihoods

# Get the background expectation value from the best-fitting point (en lieu of theoretical prediction)
expected_beta = np.array([1.10031837e+01, 8.87953673e+00, 7.61866954e+00, 6.49040517e+00, 5.73566527e+00])
expected_bkg = background_signal(expected_beta)

# Define "prior transform" for generating pseudo-data under the background-only hypothesis
def generate_pseudo_data(cube):
    return poisson.ppf(cube, mu=expected_bkg)

sigma_from_atlas = 3.9 / (2.0*np.sqrt(2.0*np.log(2.0)))
simple_sig_bfg = [130, 10]
simple_sig_bounds = [[110.,150.], [0.,600.]]
simple_bkg_bfg = [0.0]
simple_bkg_bounds = [[-0.3, 0.3]]

def guess_loc_scale(data):
    bkg = expected_bkg
    chi = (data - bkg) / bkg**0.5
    ii = chi.argmax()
    loc = bins[ii]
    signal = gaussian_signal(edges, loc, sigma_from_atlas, simple_sig_bfg[1])[ii]
    required = (data - bkg)[ii]
    scale = required / signal
    return loc, scale

rng = default_rng()

class loglike_wrapper_simple_spb(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        mu, sigma, n_events = x[1], sigma_from_atlas, x[2]
        sig = gaussian_signal(edges, mu, sigma, n_events)
        bkg = (x[0] + 1.0)*expected_bkg
        spb = sig + bkg
        return -2.0 * np.maximum(self.logpmf(spb), -1.0e99)

class loglike_wrapper_simple_bkg(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        bkg = (x[0] + 1.0)*expected_bkg
        return -2.0 * np.maximum(self.logpmf(bkg), -1.0e99)

def nested_ts_simple(data):
    xinit0 = np.array(10*[simple_bkg_bfg] + [[rng.uniform(x[0],x[1]) for x in simple_bkg_bounds] for i in range(40)])
    res0 = differential_evolution(loglike_wrapper_simple_bkg(data), bounds=simple_bkg_bounds, init=xinit0, tol=0.0001)
    ts0 = res0.fun
    x0 = list(res0.x)+simple_sig_bfg
    x0[-2:] = guess_loc_scale(data)
    x00 = np.copy(x0)
    x00[-1] = 0.0
    xlims = simple_bkg_bounds+simple_sig_bounds
    xinit = np.array(20*[x0] + 20*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(110)])
    res1 = differential_evolution(loglike_wrapper_simple_spb(data), bounds=xlims, init=xinit, tol=0.0001)
    ts1 = res1.fun
    return ts0-ts1

def nested_ts_simple_fast(data):
    res0 = minimize(loglike_wrapper_simple_bkg(data), x0=simple_bkg_bfg, bounds=simple_bkg_bounds)
    ts0 = res0.fun
    x0 = list(res0.x)+simple_sig_bfg
    x0[-2:] = guess_loc_scale(data)
    x00 = np.copy(x0)
    x00[-1] = 0.0
    xlims = simple_bkg_bounds+simple_sig_bounds
    xinit = np.array(10*[x0] + 10*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(80)])
    res1 = differential_evolution(loglike_wrapper_simple_spb(data), bounds=xlims, init=xinit, tol=0.0001)
    ts1 = res1.fun
    return ts0-ts1
