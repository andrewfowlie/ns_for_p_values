from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

import higgs_functions as higgs

n_sims = int(3)

# Run the analysis (takes ~ 10 mins)
ts_vals = np.array([higgs.calculate_ts(rvs) for rvs in tqdm(poisson.rvs(n_sims*[higgs.expected_bkg]))])
print(ts_vals)
ts_obs = higgs.calculate_ts(higgs.counts)
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
