"""
Fit Higgs to digamma data
=========================
"""

import os
import numpy as np
from scipy.special import binom, xlogy
from scipy.optimize import minimize

# Digitised data from Fig. 4 in [arXiv:1207.7214]

script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(script_dir+"/atlas_higgs_digamma_data.dat", unpack=True)
edges = np.linspace(100, 160, num=31, endpoint=True)
center = np.linspace(100, 160, num=30, endpoint=True)
nbins = 30
sigma_from_atlas = 3.9 / (2.0*np.sqrt(2.0*np.log(2.0)))

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

def fixed_bernstein_basis_ediff1d(x, k):
    basis = fixed_bernstein_basis(x, k)
    return np.maximum(basis[1:, :] - basis[:-1, :], 0.)

# Compute the basis once and for all at the bin edges
bb = fixed_bernstein_basis_ediff1d(edges, 5)

class memoized_background_events(object):
    def __init__(self):
        self.cache = None
        self.shape = None
    def __call__(self, beta):
        if not np.array_equal(beta[1:], self.cache):
            self.cache = beta[1:]
            self.shape = bb[:, 0] + np.matmul(bb[:, 1:], beta[1:])

        return beta[0] * self.shape

background_events = memoized_background_events()

# Functions for modelling signal

f = 2. / np.sqrt(2. * np.pi * sigma_from_atlas)
center_over_sigma = center / sigma_from_atlas

class memoized_gaussian_signal(object):
    def __init__(self):
        self.cache = {}
    def __call__(self, x):

        round_x = round(x[0], 2)

        try:
            shape = self.cache[round_x]
        except KeyError:
            z = center_over_sigma - round_x / sigma_from_atlas
            shape = self.cache[round_x] = f * np.exp(-0.5 * z**2)

        return x[1] * shape

gaussian_signal = memoized_gaussian_signal()

# Log-likelihoods

# Get the background expectation value from the best-fitting point (en lieu of theoretical prediction)
bkg_expected = np.array([11.0031837, 0.80699704, 0.69240592, 0.58986611, 0.52127324])
bkg_events = background_events(bkg_expected)
bkg_events_sum = bkg_events.sum()

def nan2num(x):
    if np.isinf(x) or np.isnan(x):
        return -1e99
    return x

class loglike(object):
    def __call__(self, x):
        expected = self.events(x)
        ll = xlogy(self.data, expected) - expected
        return -2. * nan2num(ll.sum() - self.norm)

class loglike_wrapper_fixed_bkg(loglike):
    def __init__(self, bkg, data):
        self.bkg = bkg
        self.data = data
        self.norm = (xlogy(self.data, self.data) - self.data).sum()

    def events(self, x):
        return self.bkg + gaussian_signal(x)

class loglike_wrapper_spb(loglike):
    def __init__(self, data):
        self.data = data
        self.norm = (xlogy(self.data, self.data) - self.data).sum()

    @staticmethod
    def events(x):
        return background_events(x[:5]) + gaussian_signal(x[5:])

class loglike_wrapper_bkg(loglike):
    def __init__(self, data):
        self.data = data
        self.norm = (xlogy(self.data, self.data) - self.data).sum()

    @staticmethod
    def events(x):
        return background_events(x)

signals = [gaussian_signal([c, 1]) for c in center]

def guess_scale(ii, data, bkg):
    d = data[ii] - bkg[ii]
    scale = d / signals[ii][ii]
    return max(0., scale)

def signal_raster_fixed_bkg(data, beta, n=5):

    bkg = background_events(beta)
    best = None

    chi = (data - bkg) / bkg**0.5
    order = np.argsort(chi)

    for ii in order[-n:]:
        mass = center[ii]
        scale = guess_scale(ii, data, bkg)

        if scale == 0.:
            continue

        bounds = ((mass - 1., mass + 1.), (0., 2. * scale))
        guess = (mass, scale)
        local = minimize(loglike_wrapper_fixed_bkg(bkg, data), guess, bounds=bounds, method="Powell",
                         options={'xtol': 1., 'ftol': 1e-4})

        if best is None or local.fun < best.fun:
            best = local

    return best

def nested_ts(data):

    tol = {'xatol': 1.,'fatol': 1e-5, 'adaptive': True}
    bkg_guess = bkg_expected
    bkg_guess[0] = data.sum() / bkg_events_sum
    polished_0 = minimize(loglike_wrapper_bkg(data), bkg_guess, method="Nelder-Mead", options=tol)
    bkg_fit = polished_0.x

    raster = signal_raster_fixed_bkg(data, bkg_fit)
    guess = bkg_fit.tolist() + raster.x.tolist()
    polished_1 = minimize(loglike_wrapper_spb(data), guess, method="Nelder-Mead", options=tol)

    return polished_0.fun - polished_1.fun

if __name__ == "__main__":
    import time
    b = bkg_expected
    s = [112.45, 137.]

    t = time.time()
    data = background_events(b) + gaussian_signal(s)
    r = nested_ts(data)
    t = time.time() - t

    print("time", t)
    print(r, b, s)
