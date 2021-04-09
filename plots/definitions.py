import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{grffile}\DeclareUnicodeCharacter{2212}{-}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
plt.rc('font', **{'family':'serif','size':10})
plt.rc('axes', labelsize=10)
plt.rc('xtick', **{'labelsize':10, 'major.pad':5})
plt.rc('ytick', **{'labelsize':10, 'major.pad':5})
plt.rc('legend', **{'fontsize':8, 'title_fontsize':10})
plt.rc('figure', titlesize=12)
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

def log10_special_formatter(x, pos):
    res = "$10^{%g}$" % (x)
    if np.abs(x) < 3:
        res = "$%g$" % (10.0**x)
    return res

def set_axis_formatter(ax, axxrange=None, axyrange=None):
    formatter = plt.FuncFormatter(log10_special_formatter)
    if (axxrange.any() != None):
        major_locator = plt.FixedLocator(axxrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(1,11)) for j in axxrange])
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(formatter)
    if (axyrange.any() != None):
        major_locator = plt.FixedLocator(axyrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(1,11)) for j in axyrange])
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_major_formatter(formatter)
