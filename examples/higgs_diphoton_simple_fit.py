import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import differential_evolution, minimize

import higgs_functions as higgs

# Wrapper for a Poisson likelihood
def wrapper_poisson_var(x, params):
    data = params
    # beta = x[4:]
    # s = np.array([higgs.crystal_ball_signal(e, e+2.0, x[0], sigma_from_atlas, x[1], x[2], x[3]) for e in higgs.bins])
    s = np.array([higgs.gaussian_signal(e, e+2.0, [x[0], sigma_from_atlas, x[1]]) for e in higgs.bins])
    b = np.array([higgs.background_signal(e, e+2.0, x[2:]) for e in higgs.bins])
    spb = np.maximum(b,0)+s
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

def wrapper_poisson_bkg_only(x, params):
    data = params
    b = np.maximum( 0, np.array([higgs.background_signal(e, e+2.0, x) for e in higgs.bins]) )
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,b)]
    return -2.0*sum(lls)

#bfb0 = 5*[[-100.,100.]]
## bfb = [[115.,135.], [0.,1.0e4], [0.,5.], [1.,500.]]+bfb0
#bfb = [[115.,135.], [0.,1.0e4]]+bfb0
#bfg0 = 5*[1.0]
#bfg = [126.0, 1.0]+bfg0
#res = differential_evolution(wrapper_poisson_var, bounds=bfb, args=[higgs.counts], popsize=100, tol=0.01)
## res = minimize(wrapper_poisson_var, x0=bfg, bounds=bfb, args=higgs.counts)
#print(res.x, res.fun) # [126.48998858 268.29987541  11.00167287   8.86230173   7.65686474 6.49888413   5.74539866] 299.9469354457792
#res0 = differential_evolution(wrapper_poisson_bkg_only, bounds=bfb0, args=[higgs.counts], popsize=100, tol=0.01)
## res0 = minimize(wrapper_poisson_bkg_only, x0=bfg0, bounds=bfb0, args=higgs.counts)
#print(res0.x, res0.fun) # [10.99229613  8.81669756  7.76544237  6.52131295  5.77230196] 309.5869059837114

# Calculate prediction from bkg-only and sig+bkg fits (parameter values from fits above)
beta = np.array([1.10018679e+01, 8.86330099e+00, 7.65461842e+00, 6.49847268e+00, 5.74487206e+00])
theta = np.array([126.48998858, 268.29987541])
bkg = np.array([higgs.background_signal(e, e+2.0,beta) for e in higgs.bins])
# sig = np.array([higgs.crystal_ball_signal(e, e+2.0, theta[0], 1.6, theta[1], theta[2], theta[3]) for e in higgs.bins])
sig = np.array([higgs.gaussian_signal(e, e+2.0, [theta[0], higgs.sigma_from_atlas, theta[1]]) for e in higgs.bins])

plt.errorbar(higgs.inv_mass, higgs.counts, yerr=np.sqrt(higgs.counts), c='k', marker='o', ls='none', label='Data')
plt.plot(higgs.inv_mass, bkg, 'r--', label='Background only')
plt.plot(higgs.inv_mass, bkg+sig, 'r-', label='Signal+background')
plt.ylim([0,4000])
plt.legend(frameon=False)
plt.xlabel('Invariant mass $m_{\gamma\gamma}$ [GeV]')
plt.ylabel('Events / 2 GeV')
plt.savefig("fitted_higgs_data.pdf")
