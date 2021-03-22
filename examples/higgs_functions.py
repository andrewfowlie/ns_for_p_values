import numpy as np
from scipy.special import binom, erf

# Digitised data from Fig. 4 in [arXiv:1207.7214]
inv_mass, counts, err_counts = np.genfromtxt("atlas_higgs_digamma_data.dat", unpack=True)

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
    spb = np.maximum(b,0)+s
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

# Wrapper for a Poisson likelihood
def wrapper_gaussian(x, params):
    data = params
    err = np.sqrt(data)
    beta = x[4:]
    s = np.array([crystal_ball_signal(e, e+2.0, x[0], x[1], x[2], 2, x[3]) for e in bins])
    b = np.array([background_signal(e, e+2.0, beta) for e in bins])
    spb = np.maximum(b,0)+s
    lls = [((k-mu)/e)**2 for (k,mu,e) in zip(data,spb,err)]
    return sum(lls)
