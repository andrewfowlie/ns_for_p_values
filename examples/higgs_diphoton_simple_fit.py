import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

import higgs_functions as higgs

# Fiting the functions to the data (takes about 5--6 mins)
## res = differential_evolution(higgs.wrapper_gaussian, bounds=[[115.,135.], [0.1,5.], [0.,1.0e4], [1.,500.]]+5*[[0,100.]], args=[higgs.counts], popsize=100, tol=0.01)
## print(res.x, res.fun)
#res = differential_evolution(higgs.wrapper_poisson, bounds=[[115.,135.], [0.1,5.], [0.,1.0e4], [1.,500.]]+5*[[0,100.]], args=[higgs.counts], popsize=100, tol=0.01)
#print(res.x, res.fun)

# Calculate prediction from bkg-only and sig+bkg fits (parameter values from fits above)
beta = np.array([1.10031837e+01, 8.87953673e+00, 7.61866954e+00, 6.49040517e+00, 5.73566527e+00])
theta = [1.26589234e+02, 2.26471587e+00, 5.20566205e+03, 3.66115121e+02]
bkg = np.array([higgs.background_signal(e, e+2.0,beta) for e in higgs.bins])
sig = np.array([higgs.crystal_ball_signal(e, e+2.0, theta[0], theta[1], theta[2], 2, theta[3]) for e in higgs.bins])

plt.errorbar(higgs.inv_mass, higgs.counts, yerr=np.sqrt(higgs.counts), c='k', marker='o', ls='none', label='Data')
plt.plot(higgs.inv_mass, bkg, 'r--', label='Background only')
plt.plot(higgs.inv_mass, bkg+sig, 'r-', label='Signal+background')
plt.ylim([0,4000])
plt.legend(frameon=False)
plt.xlabel('Invariant mass $m_{\gamma\gamma}$ [GeV]')
plt.ylabel('Events / 2 GeV')
plt.savefig("fitted_higgs_data.pdf")
