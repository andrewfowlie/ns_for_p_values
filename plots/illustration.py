"""
Sketch of method.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats

np.random.seed(185)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# define model - chi-squared with particular number of dofs
model = scipy.stats.chi2(3)

# define tail region of interest
chi2_critical = 8.26
p_value = model.sf(chi2_critical)

# plot pdf etc

chi2 = np.linspace(0, 15, 10000)
tail = chi2 > chi2_critical
pdf = model.pdf(chi2)

for a in ax:
    a.set_xlabel("$\chi^2$")
    a.set_xlim(0, 15)
    a.plot(chi2, pdf, color="SeaGreen", lw=3)
    # shade tail region
    a.fill_between(chi2, pdf, where=tail, color="Crimson", alpha=0.6, linewidth=0, zorder=-10, label="Desired tail area")
    # show critical
    a.vlines(chi2_critical, 0, model.pdf(chi2_critical), color="Crimson", lw=3, label="Critical $\chi^2$")

ax[0].set_ylabel("$p(\chi^2)$")
ax[0].set_ylim(0, None)
ax[0].set_yticks([])

# show thresholds - make a list of them from uniform compression

n = 6
alpha = np.linspace(0.3, 1., n).tolist()
x_start = model.sf(0.3)
compression = p_value / x_start
t = compression**(1. / n)
nlive = -1. / np.log(t)
chi2_threshold = [model.isf(x_start * t**i) for i in range(0, n + 1)]
print(nlive)

for i, (t, a) in enumerate(zip(chi2_threshold, alpha)):
    ax[1].vlines(t, 0, model.pdf(t), color="goldenrod", lw=3, alpha=a, label="Threshold $\chi^2$" if not i else None)

# show arrows and shading indicating increasing threshold

for i in range(n):
    dx = 0.03
    x1 = chi2_threshold[i] - dx
    x2 = chi2_threshold[i + 1] + dx
    y = 0.4 * model.pdf(x1) if x1 > 1. else 0.6 * model.pdf(1.)
    arrow = mpatches.FancyArrowPatch((x1, y), (x2, y), mutation_scale=10, color="goldenrod", lw=0, zorder=15, alpha=alpha[i])
    ax[1].add_patch(arrow)

    where = np.logical_and(chi2 > x1, chi2 < x2)
    ax[1].fill_between(chi2, pdf, where=where, alpha=alpha[i],
                       color="Gold", linewidth=0, zorder=-5,
                       label="We draw from $\chi^2 >$ threshold" if not i else None)

# annotations

ax[0].text(0.65, 0.4, 'Estimate tail area by fraction\nof samples that fall into it',
           fontsize=8, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)

ax[1].text(9.5, 0.12, 'At each iteration, threshold increases\nand yellow area compresses by factor $e^{-1 / n}$\n\nEstimate tail area by compression when\nthreshold reaches critical value',
           fontsize=8, horizontalalignment='center', verticalalignment='center')

# show random draws

# MC sampling

mc_draws = model.rvs(size=50)
for i, r in enumerate(mc_draws):
    ax[0].axvline(r, ymax=0.03, color="black", label="50 random draws" if not i else None)

# Nested sampling

mc_draws = []
while len(mc_draws) < 50:
    r = model.rvs()
    if r >= chi2_threshold[0]:
        mc_draws.append(r)

for i, r in enumerate(mc_draws):
    ax[1].axvline(r, ymax=0.03, color="black", label="50 random draws above threshold" if not i else None, zorder=100)

# finish up

for a in ax:
    a.legend(fontsize=8)
    a.set_ylim(0, None)

ax[0].set_title("Monte Carlo")
ax[1].set_title("Nested sampling")

plt.tight_layout()
plt.savefig("ill.pdf")
