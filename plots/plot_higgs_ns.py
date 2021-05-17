"""
Illustration of method on Higgs resonance search like problem
=============================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

sys.path.append("./..")
from p_value.ns import pc, ns_result

def get_line(file, label):
    with open(file) as res_file:
        for line in res_file:
            if line.strip() == label:
                next_line = res_file.readline()
                return np.array([float(e) for e in next_line.split()])
        else:
            raise RuntimeError("didn't find {}".format(label))

def req_n_mc(p, log10err):
    return 1.0 / ((log10err*np.log(10))**2 * p)

# Analyse MC results
higgs_ts_samples = np.genfromtxt("all_tsvals_higgs.dat")

hts_sorted = np.sort(higgs_ts_samples)
lhtss = float(len(higgs_ts_samples))
pvals_range = []
i = 1
ts_vals = np.linspace(0,50,250)
for ts in np.flip(ts_vals):
    while (hts_sorted[-i] > ts) and (i < len(hts_sorted)):
        i += 1
    pvals_range.append((i-1)/lhtss)

pvals_range = np.flip(pvals_range)
log10pval_errs_range = np.array([np.sqrt(1/(p*lhtss))/np.log(10.) for p in pvals_range])

# Analyse PCh results
n_live = 100
root = "pc_higgs"
label1 = "=== local volume -- log(<X_p>) ==="
label2 = "=== Number of likelihood calls ==="

log_xp = get_line("chains/"+root+".resume", label1)
log_x = logsumexp(log_xp)
n_iter = -log_x * n_live

ev_name = "chains/"+root+"_dead.txt"
ev_data = np.genfromtxt(ev_name)
test_statistic = ev_data[:, 0]
log_x = -np.arange(0, len(test_statistic), 1.) / n_live
log_x_delta = np.sqrt(-log_x / n_live)

pch_calls = int(get_line("chains/"+root+".resume", label2)[0])

res = ns_result(n_iter, n_live, pch_calls)

print(res.log10_pvalue, res.error_log10_pvalue, pch_calls, round(req_n_mc(10**res.log10_pvalue, res.error_log10_pvalue)), 1/10**res.log10_pvalue)


fig, ax = plt.subplots()

ax.plot(test_statistic, np.log10(np.exp(log_x)), c='grey', label='Polychord')
ax.fill_between(test_statistic, log_x*np.log10(np.e)-log_x_delta/np.log(10), log_x*np.log10(np.e)+log_x_delta/np.log(10), fc='grey', alpha=0.6)
ax.plot(ts_vals, np.log10(pvals_range), c='k', ls=':', label='MC')
ax.fill_between(ts_vals, np.log10(pvals_range)-log10pval_errs_range, np.log10(pvals_range)+log10pval_errs_range, fc='k', alpha=0.6)

plt.xlabel('TS')
plt.ylabel('$\mathrm{log}_{10}(p)$')
plt.legend(title='Higgs example')
plt.xlim([0,50])
plt.ylim([-6,0])
plt.savefig('pch_vs_mc.pdf')
plt.show()
