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

def logpmf(k, mu, gammaln_sum):
    ll = xlogy(k, mu) -  mu
    return ll.sum() - gammaln_sum

# Digitised data from Fig. 4 in [arXiv:1207.7214]

script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(script_dir+"/atlas_higgs_digamma_data.dat", unpack=True)
bins = np.linspace(100, 160, num=30, endpoint=False)
edges = bins.tolist() + [bins[-1] + 2.]

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

# Functions for modelling signal

def crystal_ball_norm(alpha, p):
    c = p*np.exp(-0.5*alpha*alpha)/((p-1)*np.abs(alpha))
    d = np.sqrt(0.5*np.pi)*(1 + erf(0.5*np.abs(alpha)))
    return c+d

def crystal_ball_density(x, mu, sigma, alpha, p, n_events):
    z = (x-mu)/sigma
    n = n_events/(sigma*crystal_ball_norm(alpha, p))
    if (z > -alpha):
        return n*np.exp(-0.5*z*z)
    else:
        y = p/np.abs(alpha)
        a = y**p * np.exp(-0.5*alpha*alpha)
        b = y - np.abs(alpha)
        return n*a/((b-z)**p)

def crystal_ball_signal(x1, x2, mu, sigma, alpha, p, n_events):
    xsep = mu - sigma*alpha
    n = n_events/(sigma*crystal_ball_norm(alpha, p))
    z11, z12 = (max(x1,xsep)-mu)/(np.sqrt(2.0)*sigma), (max(x2,xsep)-mu)/(np.sqrt(2.0)*sigma)
    x21, x22 = min(x1,xsep), min(x1,xsep)
    res = 0
    if (z12 > z11):
        res += np.sqrt(0.5*np.pi)*sigma*(erf(z12)-erf(z11))
    if (x22 > x21):
        y = p/np.abs(alpha)
        a = y**p * np.exp(-0.5*alpha*alpha)
        b = y - np.abs(alpha)
        res += a*(-sigma)**p * ((x22-mu-b*sigma)**(1-p) - (x21-mu-b*sigma)**(1-p)) / (p-1)
    return n*res

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

class loglike_wrapper_spb(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        # For now: just use the Gaussian instead of the crystal
        mu, sigma, n_events = x[5:]
        beta = x[:5]
        sig = gaussian_signal(edges, mu, sigma, n_events)
        bkg = background_signal(beta)
        spb = sig + bkg
        return -2. * np.maximum(self.logpmf(spb), -1.0e99)

class loglike_wrapper_bkg(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        beta = x[:5]
        bkg = background_signal(beta)
        return -2. * np.maximum(self.logpmf(bkg), -1.0e99)

sigma_from_atlas = 3.9 / (2.0*np.sqrt(2.0*np.log(2.0)))
bkg_bfg = [11., 8.8, 7.6, 6.5, 5.7]
bkg_bounds = 5*[[0.,75.]]
sig_bfg = [126, sigma_from_atlas, 1]
sig_bounds = [[110.,150.], [1.0,6.0], [0.,1000.]]
red_sig_bfg = [126, 1.0]
red_sig_bounds = [[110.,150.], [0.,500.]]

def calculate_ts():
    data = poisson.rvs(expected_bkg)
    res0 = minimize(loglike_wrapper_bkg(data), x0=bkg_bfg, bounds=bkg_bounds)
    #res0 = differential_evolution(loglike_wrapper_bkg, bounds=bkg_bounds, args=(data,), popsize=25, tol=0.01)
    ts0 = res0.fun
    x0 = list(res0.x)+sig_bfg
    # res1 = minimize(loglike_wrapper_spb(data), x0=x0, bounds=bkg_bounds+sig_bounds)
    res1 = differential_evolution(loglike_wrapper_spb, bounds=bkg_bounds+sig_bounds, args=(data,), popsize=25, tol=0.01)
    ts1 = res1.fun
    res = np.concatenate((res1.x, [ts0-ts1]))
    return np.array(res)

class loglike_wrapper_red_spb(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        mu, sigma, n_events = x[5], sigma_from_atlas, x[6]
        beta = x[:5]
        sig = gaussian_signal(edges, mu, sigma, n_events)
        bkg = background_signal(beta)
        spb = sig + bkg
        return -2. * np.maximum(self.logpmf(spb), -1.0e99)

class loglike_wrapper_red_bkg(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        beta = x[:5]
        bkg = background_signal(x)
        return -2. * np.maximum(self.logpmf(bkg), -1.0e99)

def guess_loc_scale(data):
    bkg = background_signal(bkg_bfg)
    chi = (data - bkg) / bkg**0.5
    ii = chi.argmax()
    loc = bins[ii]
    signal = gaussian_signal(edges, loc, sigma_from_atlas, red_sig_bfg[1])[ii]
    required = (data - bkg)[ii]
    scale = required / signal
    return loc, scale

def nested_ts(data):
    rng = default_rng()
    # res0 = minimize(loglike_wrapper_red_bkg(data), x0=bkg_bfg, bounds=bkg_bounds)
    # res0 = differential_evolution(loglike_wrapper_red_bkg(data), bounds=bkg_bounds, args=(data,), popsize=25, tol=0.01)
    xinit0 = np.array(20*[bkg_bfg] + [[rng.uniform(x[0],x[1]) for x in bkg_bounds] for i in range(80)])
    res0 = differential_evolution(loglike_wrapper_red_bkg(data), bounds=bkg_bounds, init=xinit0, tol=0.0001)
    ts0 = res0.fun
    x0 = list(res0.x)+red_sig_bfg
    x0[-2:] = guess_loc_scale(data)
    x00 = np.copy(x0)
    x00[-1] = 0.0
    # res1 = minimize(loglike_wrapper_red_spb(data), x0=x0, bounds=bkg_bounds+red_sig_bounds)
    # res1 = differential_evolution(loglike_wrapper_red_spb(data), bounds=bkg_bounds+red_sig_bounds, popsize=75, tol=0.005)
    xlims = bkg_bounds+red_sig_bounds
    xinit = np.array(30*[x0] + 30*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(140)])
    res1 = differential_evolution(loglike_wrapper_red_spb(data), bounds=xlims, init=xinit, tol=0.0001)
    ts1 = res1.fun
    return ts0-ts1

class loglike_wrapper_minimal_spb(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        bkg = background_signal(expected_beta)
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        mu, sigma, n_events = x[0], sigma_from_atlas, x[1]
        sig = gaussian_signal(edges, mu, sigma, n_events)
        spb = sig + self.bkg
        return -2. * np.maximum(self.logpmf(spb), -1.0e99)

def nested_ts_bkg(data):
    llwrap = loglike_wrapper_red_bkg(data)
    ts0 = llwrap(expected_beta)
    x0 = guess_loc_scale(data)
    x00 = np.copy(x0)
    x00[-1] = 0.0
    xlims = red_sig_bounds
    xinit = np.array(20*[x0] + 20*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(60)])
    res1 = differential_evolution(loglike_wrapper_minimal_spb(data), bounds=xlims, init=xinit, tol=0.0001)
    ts1 = res1.fun
    return ts0-ts1

def nested_ts_full(data):
    rng = default_rng()
    # res0 = minimize(loglike_wrapper_red_bkg(data), x0=bkg_bfg, bounds=bkg_bounds)
    # res0 = differential_evolution(loglike_wrapper_red_bkg(data), bounds=bkg_bounds, args=(data,), popsize=25, tol=0.01)
    xinit0 = np.array(20*[bkg_bfg] + [[rng.uniform(x[0],x[1]) for x in bkg_bounds] for i in range(130)])
    res0 = differential_evolution(loglike_wrapper_red_bkg(data), bounds=bkg_bounds, init=xinit0, tol=0.001)
    ts0 = res0.fun
    x0 = list(res0.x)+sig_bfg
    x00 = np.copy(x0)
    x00[-1] = 0.0
    # res1 = minimize(loglike_wrapper_red_spb(data), x0=x0, bounds=bkg_bounds+red_sig_bounds)
    # res1 = differential_evolution(loglike_wrapper_red_spb(data), bounds=bkg_bounds+red_sig_bounds, popsize=75, tol=0.005)
    xlims = bkg_bounds+sig_bounds
    xinit = np.array(50*[x0] + 50*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(100)])
    res1 = differential_evolution(loglike_wrapper_red_spb(data), bounds=xlims, init=xinit, tol=0.001)
    ts1 = res1.fun
    return ts0-ts1

class loglike_wrapper_simple_spb(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        mu, sigma, n_events = x[5], sigma_from_atlas, x[6]
        beta = x[:5]
        sig = gaussian_signal(edges, mu, sigma, n_events)
        bkg = (x[0] + 1.0)*expected_bkg
        spb = sig + bkg
        return -2. * np.maximum(self.logpmf(spb), -1.0e99)

class loglike_wrapper_simple_bkg(object):
    def __init__(self, data):
        gammaln_sum = gammaln(data + 1).sum()
        self.logpmf = lambda x: logpmf(data, x, gammaln_sum)
    def __call__(self, x):
        bkg = (x[0] + 1.0)*expected_bkg
        return -2. * np.maximum(self.logpmf(bkg), -1.0e99)

simple_bkg_bfg = [0.0]
simple_bkg_bounds = [[-0.5, 1.0]]

def nested_ts_simple(data):
    rng = default_rng()
    xinit0 = np.array(10*[simple_bkg_bfg] + [[rng.uniform(x[0],x[1]) for x in simple_bkg_bounds] for i in range(40)])
    res0 = differential_evolution(loglike_wrapper_simple_bkg(data), bounds=bkg_bounds, init=xinit0, tol=0.0001)
    ts0 = res0.fun
    x0 = list(res0.x)+red_sig_bfg
    x0[-2:] = guess_loc_scale(data)
    x00 = np.copy(x0)
    x00[-1] = 0.0
    xlims = simple_bkg_bounds+red_sig_bounds
    xinit = np.array(20*[x0] + 20*[x00] + [[rng.uniform(x[0],x[1]) for x in xlims] for i in range(110)])
    res1 = differential_evolution(loglike_wrapper_simple_spb(data), bounds=xlims, init=xinit, tol=0.0001)
    ts1 = res1.fun
    return ts0-ts1
