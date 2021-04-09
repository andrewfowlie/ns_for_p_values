"""
Sketch of method.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import lines
import numpy as np
import scipy.stats

import definitions

np.random.seed(185)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.75), sharey=True)

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
    a.set_xlabel("Test statistic $\chi^2$")
    a.set_xlim(0, 15)
    a.plot(chi2, pdf, color="RoyalBlue", lw=3)
    # shade tail region
    a.fill_between(chi2, pdf, where=tail, color="Crimson", alpha=0.6, linewidth=0, zorder=-10)#, label="Desired tail area")
    # show critical
    a.vlines(chi2_critical, 0, model.pdf(chi2_critical), color="Crimson", lw=3)#, label="Critical $\chi^2$")

ax[0].set_ylabel(r"$\mathrm{PDF}(\chi^2)$")
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
    ax[1].vlines(t, 0, model.pdf(t), color="goldenrod", lw=3, alpha=a) # , label="Threshold $\chi^2$" if not i else None)

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
                       color="Gold", linewidth=0, zorder=-5)#,
                       #label="We draw from $\chi^2 >$ threshold" if not i else None)

# annotations

ax[0].text(0.74, 0.15,
           'Tail area$\\approx$fraction of\n'
           'samples above\n'
           'critical value',
           fontsize=8, color="Crimson", horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)

ax[1].text(0.35, 0.62, 'At each iteration:',
           fontsize=8, horizontalalignment='left', verticalalignment='top', transform=ax[1].transAxes)

ax[1].text(0.35, 0.56,
           ' $\\bullet$ threshold increases\n'
           ' $\\bullet$ draw from $\chi^2 >$ threshold\n'
           ' $\\bullet$ yellow area compresses by$\\approx\mathrm{e}^{-1 / n}$',
           fontsize=8, horizontalalignment='left', verticalalignment='top', transform=ax[1].transAxes)

arrow = mpatches.FancyArrowPatch((0.34, 0.605), (0.2, 0.465), mutation_scale=10, color="k", lw=0, zorder=15, transform=ax[1].transAxes)
ax[1].add_patch(arrow)

ax[1].text(0.74, 0.15,
           'Tail area$\\approx$yellow area\n'
           'when threshold reaches\n'
           'critical value',
           fontsize=8, color="Crimson", horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)

# show random draws

# MC sampling

mc_draws = model.rvs(size=50)
for i, r in enumerate(mc_draws):
    ax[0].axvline(r, ymax=0.03, color="darkgrey")#, label="50 random draws" if not i else None)

# Nested sampling

mc_draws = []
while len(mc_draws) < 50:
    r = model.rvs()
    if r >= chi2_threshold[0]:
        mc_draws.append(r)

# for i, r in enumerate(mc_draws):
#     ax[1].axvline(r, ymax=0.03, color="black", label="50 random draws above threshold" if not i else None, zorder=100)

# finish up

for a in ax:
    # a.legend(fontsize=8)
    a.set_ylim(0, None)

# custom legend markers

def vertical_line(color, width, height):
    return lines.Line2D([], [],  marker='|', linestyle='None', color=color, markersize=height, markeredgewidth=width)

def add_vertical_line(handles, labels, label, color, width=1.5, height=8):
    handles.append(vertical_line(color, width, height))
    labels.append(label)

handles, labels = ax[1].get_legend_handles_labels()
add_vertical_line(handles, labels, "Threshold $\chi^2$", "goldenrod", 3, 12)
add_vertical_line(handles, labels, "Critical $\chi^2$", "Crimson", 3, 12)
ax[1].legend(handles, labels, handletextpad=0.1, ncol=2, columnspacing=0.5)

handles, labels = ax[0].get_legend_handles_labels()
add_vertical_line(handles, labels, "50 random draws", "darkgrey")
add_vertical_line(handles, labels, "Critical $\chi^2$", "Crimson", 3, 12)
ax[0].legend(handles, labels, handletextpad=0.1, ncol=2, columnspacing=0.5)

ax[0].set_title("Monte Carlo")
ax[1].set_title("Nested sampling")

plt.tight_layout()
plt.savefig("ill.pdf", backend='pgf')
