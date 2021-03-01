import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom, erf
from scipy.optimize import differential_evolution

# Digitised data from Fig. 4 in [arXiv:1207.7214]
inv_mass, counts, err_counts = np.genfromtxt("atlas_higgs_digamma_data.dat", unpack=True)

# Berstein polynomials for background model
def bernstein_basis_poly(x, nu, k):
    return binom(k, nu) * x**nu * (1 - x)**(k - nu)

def bernstein_poly(x, beta):
    k = len(beta)-1
    return sum([beta[nu]*bernstein_basis_poly(x, nu, k) for nu in range(k+1)])

# Crystal Ball function for the signal model (scaled with nn = number of events)
def crystal_ball(x, mu, sigma, alpha, p, nn):
    z = (x-mu)/sigma
    c = p*np.exp(-0.5*alpha*alpha)/((p-1)*np.abs(alpha))
    d = np.sqrt(0.5*np.pi)*(1 + erf(0.5*np.abs(alpha)))
    n = nn/(sigma*(c+d))
    if (z > -alpha):
        return n*np.exp(-0.5*z*z)
    else:
        y = p/np.abs(alpha)
        a = y**p * np.exp(-0.5*alpha*alpha)
        b = y - np.abs(alpha)
        return n*a/((b-z)**p)

# Wrapper to fit the data.
# N.B. A proper fit to the data should calculate the signal from the integrals and use a Poisson likelihood
def wrapper_de_approx(x, m, d, e):
    return sum([((d[i]-crystal_ball_pdf(m[i], x[0], x[1], x[2], 2, x[3])-bernstein_poly((m[i]-100.)/(160.-100.), beta=x[4:]))/e[i])**2 for i in range(len(d))])

# Fit the functions to the data
#res = differential_evolution(wrapper_de2, bounds=[[120,130], [0.5,5], [-100,100], [1,1.0e3]]+5*[[-1e4,1e4]], args=[inv_mass_data,counts,np.sqrt(counts)], popsize=200, tol=0.005)
#print(res.x, res.fun)

# Calculate prediction from bkg-only and sig+bkg fits
imasses = [100+0.5*i for i in range(120)]
bkg = [bernstein_poly((m-100.)/(160.-100.), beta=[3665.43758301, 2209.88671378, 1889.185537, 927.71601952, 924.5768427]) for m in imasses]
sigplusbkg = [crystal_ball(m, 126.5, 2.34684666, 4.46742043e+01, 2, 7.46509001e+02)+bernstein_poly((m-100.)/(160.-100.), beta=[3.66944572e+03, 2.24599218e+03, 1.70411248e+03, 1.03176908e+03, 9.04769723e+02]) for m in imasses]

plt.errorbar(inv_mass, counts, yerr=np.sqrt(counts), c='k', marker='o', ls='none')
plt.plot(imasses, bkg, 'r--', label='Background only')
plt.plot(imasses, sigplusbkg, 'r-', label='Signal+background')
plt.ylim([0,4000])
plt.legend(frameon=False)
plt.xlabel('Invariant mass $m_{\gamma\gamma}$ [GeV]')
plt.ylabel('Events / 2 GeV')
plt.savefig("fitted_higgs_data.pdf")
