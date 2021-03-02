"""
Sketch of method.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import numpy as np
from scipy.stats import chi2

np.random.seed(185)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# define model - chi-squared with particular number of dofs
model = chi2(3)

# define tail region of interest
critical = 8.26
p = model.sf(critical)


# plot pdf etc

x = np.linspace(0, 15, 10000)
tail = x > critical 
y = model.pdf(x)

for a in ax:
    a.set_xlabel("$\chi^2$")
    a.set_xlim(0, 15)
    dx = 0.1
    a.plot(x, y, color="SeaGreen", lw=3)
    # shade tail region
    a.fill_between(x, y, where=tail, color="Crimson", alpha=0.6, linewidth=0, zorder=-10, label="Desired tail area")
    
ax[0].set_ylabel("$p(\chi^2)$")
ax[0].set_ylim(0, None)
ax[0].set_yticks([])

# MC sampling - show draws as small vertical lines

ax[0].set_title("Monte Carlo")

rv = model.rvs(size=50)
ax[0].axvline(rv[0], ymax=0.05, color="black", label="50 random draws")
for r in rv[1:]:
    ax[0].axvline(r, ymax=0.05, color="black")
    
# Nested sampling

ax[1].set_title("Nested sampling")

star = 1.5

agold = mcolors.colorConverter.to_rgba('Gold', alpha=.3)
where = np.logical_and(x > star, x < critical)
ax[1].fill_between(x, y, where=where, facecolor="Gold", edgecolor="Gold", alpha=0.3, linewidth=0, zorder=-5, label="We draw from $\chi^2 >$ threshold")
ax[1].axvline(star, ymax=0.05, color="black", label="50 random draws")

rv_star = []
while len(rv_star) < 50:
    r = model.rvs()
    if r > star:
        rv_star.append(r)

for r in rv_star[1:]:
    ax[1].axvline(r, ymax=0.05, color="black")

# show arrow indicating increasing threshold

x1 = star - 0.05
x2 = star + 3
y = 0.12
arrow = mpatches.FancyArrowPatch((x1, y), (x2, y), mutation_scale=10, color="goldenrod",  zorder=15)
ax[1].add_patch(arrow)

# show thresholds

for a in ax:
    a.vlines(critical, 0, model.pdf(critical), color="Crimson", lw=3, label="Critical $\chi^2$")
ax[1].vlines(star, 0, model.pdf(star), color="goldenrod", lw=3, label="Threshold $\chi^2$")

# annotations

ax[0].text(0.65, 0.4, 'Estimate tail area by fraction\nof samples that fall into it',
           fontsize=8, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)

ax[1].text(9.5, y, 'At each iteration, threshold increases\nand yellow area compresses by factor $t$\n\nEstimate tail area by compression when\nthreshold reaches critical value',
           fontsize=8, horizontalalignment='center', verticalalignment='center')


# finish up

ax[0].legend(fontsize=8)
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ill.pdf")
