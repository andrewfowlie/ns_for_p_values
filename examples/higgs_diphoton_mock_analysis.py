from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import differential_evolution, minimize

import higgs_functions as higgs

n_sims = int(2e3)

# For now: just get the background expectation value from the best-fitting point
beta = np.array([1.10031837e+01, 8.87953673e+00, 7.61866954e+00, 6.49040517e+00, 5.73566527e+00])
expected_bkg = np.array([higgs.background_signal(e, e+2.0,beta) for e in higgs.bins])

def loglike_wrapper_spb(x, data):
    # For now: just get the Gaussian instead of the crystal
    sig = np.array([higgs.gaussian_signal(e, e+2.0, x) for e in higgs.bins])
    spb = sig + np.maximum(expected_bkg, 0)
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,spb)]
    return -2.0*sum(lls)

def loglike_wrapper_bkg(data):
    lls = [poisson.logpmf(k=k, mu=mu) for (k,mu) in zip(data,expected_bkg)]
    return -2.0*sum(lls)

def calculate_ts(data):
    ts0 = loglike_wrapper_bkg(data)
    temp = minimize(loglike_wrapper_spb, x0=[126, 2.5, 350], bounds=[[115.,135.], [0.1,5.], [0.,1.0e4]], args=data)
    ts1 = temp.fun
    return np.concatenate((temp.x, [ts0-ts1]))

# Run the analysis (takes ~ 10 mins)
ts_vals = np.array([calculate_ts(rvs) for rvs in tqdm(poisson.rvs(n_sims*[expected_bkg]))])
ts_obs = calculate_ts(higgs.counts)
print("Optmised observed data: ", ts_obs)
pval = sum(ts_vals[:,-1] > ts_obs[-1]) / float(n_sims)
print('P-value from MC: {:.3e}.'.format(pval))

fig, ax = plt.subplots()

ax.hist(ts_vals[:,-1])
ax.axvline(x=ts_obs[-1], c='r', label='observed')

ax.legend(frameon=False)
ax.set_xlabel('TS')
ax.set_ylabel('MC samples')
plt.savefig("example_higgs_mc.pdf")
plt.show()
