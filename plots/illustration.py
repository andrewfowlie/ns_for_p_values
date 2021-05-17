"""
Sketch of NS and MC methods
===========================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib import lines
import numpy as np
import scipy.stats

from definitions import set_style

def vertical_line(color, width, height):
    return lines.Line2D([], [], marker='|', linestyle='None', color=color, markersize=height, markeredgewidth=width)

def add_vertical_line(handles, labels, label, color, width=1.5, height=9):
    handles.append(vertical_line(color, width, height))
    labels.append(label)

def custom_text_line(axis, pos_x, pos_y, text):
    axis.text(pos_x, pos_y,
              text,
              fontsize=9, horizontalalignment='left', verticalalignment='center', transform=axis.transAxes)

np.random.seed(185)
set_style(gs=14)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.75), sharey=True)
ts_name = "\\ensuremath{\\lambda}"

# define model - chi-squared with particular number of dofs
model = scipy.stats.chi2(3)

# define tail region of interest
chi2_critical = 8.26
p_value = model.sf(chi2_critical)

# plot pdf etc

chi2 = np.linspace(0, 15, 200)
tail = chi2 > chi2_critical
pdf = model.pdf(chi2)

for a in ax:
    a.plot(chi2, pdf, color="RoyalBlue", lw=3)
    # shade tail region
    a.fill_between(chi2, pdf, where=tail, color="Crimson", alpha=0.4, linewidth=0, zorder=-10)#, label="Desired tail area")
    # show critical
    a.vlines(chi2_critical, 0, model.pdf(chi2_critical), color="Crimson", lw=3)#, label="Critical $\chi^2$")
    a.set_xlabel("Test statistic "+ts_name)
    a.set_xlim(0, 15)
    a.set_ylim(0, None)
    a.yaxis.set_ticks([])
    a.yaxis.set_ticklabels([])

# MC sampling
shade_of_grey = 'grey'
def outline(lw=2):
    return [path_effects.Stroke(linewidth=lw, foreground='white'), path_effects.Normal()]
mc_draws = model.rvs(size=50)
for i, r in enumerate(mc_draws):
    ax[0].axvline(r, ymax=0.03, color=shade_of_grey)#, path_effects=outline())#, label="50 random draws" if not i else None)

handles, labels = ax[0].get_legend_handles_labels()
# add_vertical_line(handles, labels, "50 random draws", "darkgrey")
add_vertical_line(handles, labels, "Critical "+ts_name, "Crimson", 3, 12)
ax[0].legend(handles, labels, handletextpad=0.1, ncol=2, columnspacing=0.5, fontsize=9, borderaxespad=1)

# Annotations
custom_text_line(ax[0], 0.27, 0.74, "Evaluate "+ts_name+" for $n$ sets of\nrandomly generated pseudo-data")

arrow = mpatches.FancyArrowPatch((1.7, 0.0325), (0.5, 0.01), mutation_scale=8, color=shade_of_grey, lw=0, zorder=15)
ax[0].add_patch(arrow)
arrow = mpatches.FancyArrowPatch((6.3, 0.0225), (9.35, 0.004), mutation_scale=8, color=shade_of_grey, lw=0, zorder=15, path_effects=outline(1.5))
ax[0].add_patch(arrow)
ax[0].text(1.75, 0.03,
           "Monte Carlo\nrandom samples",
           fontsize=9, color=shade_of_grey, horizontalalignment='left', verticalalignment='center')

ax[0].text(11.75, 0.035,
           'Tail area $\\approx$\n'
           'fraction of samples\n'
           'above critical value',
           fontsize=9, color="Crimson", horizontalalignment='center', verticalalignment='center')

ax[0].set_ylim(0, None)
ax[0].set_title("Monte Carlo", fontsize=10)
ax[0].set_ylabel("$\\text{PDF}("+ts_name+")$")
ax[0].tick_params(bottom=True, top=False, left=False, right=False)

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

# Annotations
arrow = mpatches.FancyArrowPatch((0.2725, 0.72), (0.2, 0.465), mutation_scale=8, color="k", lw=0, zorder=15, transform=ax[1].transAxes)
ax[1].add_patch(arrow)
custom_text_line(ax[1], 0.27, 0.74, "At each iteration:")
custom_text_line(ax[1], 0.28, 0.68, "\\textbullet threshold $"+ts_name+"^\\star$ increases")
custom_text_line(ax[1], 0.28, 0.63, "\\textbullet draw from $"+ts_name+" > "+ts_name+"^\\star$")
custom_text_line(ax[1], 0.28, 0.58, "\\textbullet area compresses by $\\sim \\mathrm{e}^{-1 / n_\\text{live}}$")


ax[1].text(0.765, 0.125,
           'Tail area $\\approx$ yellow area\n'
           'when threshold reaches\n'
           'critical '+ts_name,
           fontsize=9, color="Crimson", horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)

# Nested sampling

mc_draws = []
while len(mc_draws) < 50:
    r = model.rvs()
    if r >= chi2_threshold[0]:
        mc_draws.append(r)


handles, labels = ax[1].get_legend_handles_labels()
add_vertical_line(handles, labels, "Thresholds $"+ts_name+"^\\star$", "goldenrod", 3, 12)
add_vertical_line(handles, labels, "Critical "+ts_name, "Crimson", 3, 12)
ax[1].legend(handles, labels, handletextpad=0.1, ncol=2, columnspacing=0.5, fontsize=9, borderaxespad=1)

ax[1].set_title("Nested sampling", fontsize=10)

plt.tight_layout(pad=0.5, w_pad=2)
#plt.savefig("ill.png")
plt.savefig("ill.pdf", backend='pgf')
