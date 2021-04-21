"""
Fit Higgs to digamma data
=========================
"""

import os
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import binom, xlogy
from scipy.optimize import minimize, minimize_scalar


# Digitised data from Fig. 4 in [arXiv:1207.7214]

script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(os.path.join(script_dir, "atlas_higgs_digamma_data.dat"), unpack=True)
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
bb0 = bb[:, 0]
bb1 = bb[:, 1:]

class memoized_background_events(object):
    def __init__(self):
        self.hash = None
        self.shape = None

    def __call__(self, beta):
        hash_ = hash(beta[1:].data.tobytes())
        if not self.hash == hash_ :
            self.hash = hash_
            self.shape = bb0 + np.matmul(bb1, beta[1:])
            self.shape /= self.shape.sum()

        return beta[0] * self.shape

background_events = memoized_background_events()

# Functions for modelling signal

f = 2. / np.sqrt(2. * np.pi * sigma_from_atlas)
center_over_sigma = center / sigma_from_atlas

class memoized_gaussian_signal(object):
    def __init__(self):
        self.cache = {}
        self.cache[0.] = np.zeros_like(center)

    def __call__(self, x):

        round_x = round(x[0], 2)

        try:
            shape = self.cache[round_x]
        except KeyError:
            z = center_over_sigma - round_x / sigma_from_atlas
            shape = self.cache[round_x] = np.exp(-0.5 * z**2)

        return f * x[1] * shape

gaussian_signal = memoized_gaussian_signal()

# Log-likelihoods

# Get the background expectation value from the best-fitting point (en lieu of theoretical prediction)
bkg_expected = np.array([5.77229940e+04, 9.16047899e-01, 5.40064675e-01, 3.97623250e-01, 8.61989197e-02])
bkg_events = background_events(bkg_expected)
bkg_events_sum = bkg_events.sum()
bin_ii = 15
signal_events = gaussian_signal((center[bin_ii], 1.))

class loglike(metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data
        self.norm = (xlogy(self.data, self.data) - self.data).sum()
        super().__init__()

    def __call__(self, x):
        expected = self.events(x)
        ll = (xlogy(self.data, expected) - expected).sum()
        return -2. * (ll - self.norm)

    @abstractmethod
    def events(self, x):
        pass

class loglike_wrapper_fixed_bkg(loglike):
    def __init__(self, data, bkg):
        self.bkg = bkg
        super().__init__(data)

    def events(self, x):
        return self.bkg + gaussian_signal(x)

class loglike_wrapper_local(loglike):
    def __init__(self, data, bkg, signal):
        self.bkg = bkg
        self.signal = signal
        super().__init__(data)

    def events(self, x):
        return self.bkg + x * self.signal


class loglike_wrapper_spb(loglike):
    @staticmethod
    def events(x):
        return background_events(x[:5]) + gaussian_signal(x[5:])

class loglike_wrapper_bkg(loglike):
    @staticmethod
    def events(x):
        return background_events(x)


signals = np.array([gaussian_signal([c, 1])[i] for i, c in enumerate(center)])

def signal_raster_fixed_bkg(wrapper, data, bkg, n=5):

    best = None

    d = data - bkg
    chi = d / bkg**0.5
    index = np.argpartition(chi, -n)[-n:]


    for ii in index:
        mass = center[ii]
        scale = max(d[ii], d[ii - 1: ii + 2].sum())

        if scale <= 0.:
            continue

        bounds = ((mass - 1., mass + 1.), (0., 2. * scale))
        guess = (mass, scale)
        local = minimize(wrapper, guess, bounds=bounds, method="Powell",
                         options={'xtol': 1e-3, 'ftol': 1e-6})

        if best is None or local.fun < best.fun:
            best = local

    return best

def nested_ts_fixed_bkg(data):
    wrapper = loglike_wrapper_fixed_bkg(data, bkg_events)
    ts0 = wrapper((0., 0.))
    raster = signal_raster_fixed_bkg(wrapper, data, bkg_events)

    return ts0 - raster.fun

def nested_ts(data):
    d = data - bkg_events
    guess = max(d[bin_ii], d[bin_ii - 1: bin_ii + 2].sum())
    if guess <= 0.:
        return 0.

    wrapper = loglike_wrapper_local(data, bkg_events, signal_events)
    ts0 = wrapper(0.)
    local = minimize_scalar(wrapper, bracket=(0., 2. * guess), options={'xtol': 1.48e-03, 'maxiter': np.inf})
    return ts0 - local.fun

def nested_ts_bkg(data):

    tol = {'xatol': 1., 'fatol': 1e-5, 'adaptive': True}

    data_sum = data.sum()
    bkg_guess = np.array(bkg_expected)
    bkg_guess[0] = data_sum
    polished_0 = minimize(loglike_wrapper_bkg(data), bkg_guess, method="Nelder-Mead", options=tol)
    bkg_fit = polished_0.x

    bkg = background_events(bkg_fit)
    wrapper = loglike_wrapper_fixed_bkg(data, bkg)
    raster = signal_raster_fixed_bkg(wrapper, data, bkg)
    bkg_guess = bkg_fit.tolist()
    bkg_guess[0] = data_sum - raster.x[1]
    guess = bkg_guess + raster.x.tolist()
    polished_1 = minimize(loglike_wrapper_spb(data), guess, method="Nelder-Mead", options=tol)

    return polished_0.fun - polished_1.fun

if __name__ == "__main__":
    import time
    b = bkg_expected
    s = [center[bin_ii], 137.]

    t = time.time()
    observed = background_events(b) + gaussian_signal(s)
    r = nested_ts(observed)
    t = time.time() - t

    print("time", t)
    print(r, b, s)
