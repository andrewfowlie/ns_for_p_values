import os
import numpy as np
from scipy.special import binom, erf
from scipy.stats import poisson
from scipy.optimize import differential_evolution, minimize

# Digitised data from Fig. 4 in [arXiv:1207.7214]
script_dir = os.path.dirname(os.path.abspath(__file__))
inv_mass, counts, err_counts = np.genfromtxt(script_dir+"/atlas_higgs_digamma_data.dat", unpack=True)

# Berstein polynomials (nu,k) for background model, x in (100,160) MeV
def bernstein_basis(x, nu, k):
    # Rescale x to (0,1)
    xprime = (x-100.)/60.
    # Return with a scaling factor ~ O(observed counts)
    return 1.0e4*binom(k, nu) * xprime**nu * (1 - xprime)**(k - nu)

def bernstein_poly(x, beta):
    k = len(beta)-1
    all_polys = [beta[nu]*bernstein_basis(x, nu, k) for nu in range(k+1)]
    return sum(all_polys)

def bernstein_integral(x, beta):
    k = len(beta)-1
    return sum([sum([beta[j-1]*bernstein_basis(x, j, k+1) for j in range(nu+1,k+2)])/(k+1) for nu in range(k+1)])

def background_signal(x1, x2, beta):
    return bernstein_integral(x2, beta) - bernstein_integral(x1, beta)

# Crystal Ball function for the signal model (scaled with number of events)
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

def gaussian_signal_integral(x, mu, sigma, n_events):
    z = (x-mu)/(np.sqrt(2.0)*sigma)
    return 0.5*n_events*erf(z)

def gaussian_signal(x1, x2, params):
     mu, sigma, n_events = params
     return gaussian_signal_integral(x2, mu, sigma, n_events) - gaussian_signal_integral(x1, mu, sigma, n_events)

bins = np.linspace(100, 160, num=30, endpoint=False)

# Wrapper for a Poisson likelihood
def wrapper_poisson(x, params):
    data = params
    beta = x[4:]
    s = np.array([crystal_ball_signal(e, e+2.0, x[0], x[1], x[2], 2, x[3]) for e in bins])
    b = np.array([background_signal(e, e+2.0, beta) for e in bins])
    spb = np.maximum(b,0) + s # Expectation value > 0
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

# Wrapper for a Poisson likelihood
def wrapper_gaussian(x, params):
    data = params
    err = np.sqrt(data)
    beta = x[4:]
    s = np.array([crystal_ball_signal(e, e+2.0, x[0], x[1], x[2], 2, x[3]) for e in bins])
    b = np.array([background_signal(e, e+2.0, beta) for e in bins])
    spb = np.maximum(b,0) + s # Expectation value > 0
    lls = [((k-mu)/e)**2 for (k,mu,e) in zip(data,spb,err)]
    return sum(lls)

### Extended analysis
# Get the background expectation value from the best-fitting point (en lieu of theoretical prediction)
beta = np.array([1.10031837e+01, 8.87953673e+00, 7.61866954e+00, 6.49040517e+00, 5.73566527e+00])
expected_bkg = np.array([background_signal(e, e+2.0, beta) for e in bins])

def loglike_wrapper_spb(x, data):
    # For now: just use the Gaussian instead of the crystal
    sig = np.array([gaussian_signal(e, e+2.0, x[5:]) for e in bins])
    bkg = np.array([background_signal(e, e+2.0, x[:5]) for e in bins])
    spb = sig + np.maximum(bkg, 0)
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

def loglike_wrapper_bkg(x, data):
    bkg = np.maximum( np.array([background_signal(e, e+2.0, x) for e in bins]), 0 )
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,bkg)]
    return -2.0*sum(lls)

bkg_bfg = [11., 8.8, 7.6, 6.5, 5.7]
bkg_bounds = 5*[[-75.,75.]]
sig_bfg = [126, 1.6, 100]
sig_bounds = [[105.,155.], [1.0,6.0], [0.,1.0e4]]

def calculate_ts(task_id=0, n_batch_size=0):
    data = poisson.rvs(expected_bkg)
    res0 = minimize(loglike_wrapper_bkg, x0=bkg_bfg, bounds=bkg_bounds, args=data)
    #res0 = differential_evolution(loglike_wrapper_bkg, bounds=bkg_bounds, args=(data,), popsize=25, tol=0.01)
    ts0 = res0.fun
    res1 = minimize(loglike_wrapper_spb, x0=bkg_bfg+sig_bfg, bounds=bkg_bounds+sig_bounds, args=data)
    #res1 = differential_evolution(loglike_wrapper_spb, bounds=bkg_bounds+sig_bounds, args=(data,), popsize=25, tol=0.01)
    ts1 = res1.fun
    res = np.concatenate((res1.x, [ts0-ts1]))
    return np.array(res)

sigma_from_atlas = 3.9 / (2.0*np.sqrt(2.0*np.log(2.0)))
red_sig_bfg = [126, 1.0]
red_sig_bounds = [[105.,155.], [0.,500.]]

def loglike_wrapper_red_spb(x, data):
    sig = np.array([gaussian_signal(e, e+2.0, (x[5],sigma_from_atlas,x[6])) for e in bins])
    bkg = np.array([background_signal(e, e+2.0, x[:5]) for e in bins])
    spb = sig + np.maximum(bkg, 0)
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

def loglike_wrapper_red_bkg(x, data):
    bkg = np.maximum( np.array([background_signal(e, e+2.0, x) for e in bins]), 0 )
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,bkg)]
    return -2.0*sum(lls)

def nested_ts(data):
    res0 = minimize(loglike_wrapper_red_bkg, x0=bkg_bfg, bounds=bkg_bounds, args=data)
    # res0 = differential_evolution(loglike_wrapper_bkg, bounds=bkg_bounds, args=(data,), popsize=25, tol=0.01)
    ts0 = res0.fun
    res1 = minimize(loglike_wrapper_red_spb, x0=bkg_bfg+red_sig_bfg, bounds=bkg_bounds+red_sig_bounds, args=data)
    # res1 = differential_evolution(loglike_wrapper_spb, bounds=bkg_bounds+sig_bounds, args=(data,), popsize=25, tol=0.01)
    ts1 = res1.fun
    return ts0-ts1
