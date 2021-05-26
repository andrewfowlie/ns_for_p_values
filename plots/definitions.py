"""
Styling for all publication plots
=================================
"""

import matplotlib.pyplot as plt
import numpy as np

def set_style(gs=10, lts=10, lfs=8, lbls=10, tls=10):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
    plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{grffile}\DeclareUnicodeCharacter{2212}{-}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
    plt.rc('font', **{'family':'serif', 'size': gs})
    plt.rc('axes', labelsize=lbls)
    plt.rc('xtick', **{'labelsize': tls, 'major.pad': 0.5*tls})
    plt.rc('ytick', **{'labelsize': tls, 'major.pad': 0.5*tls})
    plt.rc('legend', **{'fontsize': lfs, 'title_fontsize': lts})
    plt.rc('figure', titlesize=12)
    plt.rc('mathtext', default='regular')

def log10_special_formatter(x, pos):
    res = "$10^{%g}$" % (x)
    if np.abs(x) < 3:
        res = "$%g$" % (10.0**x)
    return res

def log10_special_formatter_every_n(x, pos, n=2):
    res = ""
    if x%n == 0:
        res = "$10^{%g}$" % (x)
        if np.abs(x) < 3:
            res = "$%g$" % (10.0**x)
    return res

def pow10_formatter(x, pos):
    lgx = int(np.log10(x))
    if np.abs(lgx) < 3:
        return "$%g$" % (x)
    return "$10^{%g}$" % (lgx)

def set_axis_formatter(ax, axxrange=None, axyrange=None):
    formatter = plt.FuncFormatter(log10_special_formatter)
    if axxrange:
        major_locator = plt.FixedLocator(axxrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(1, 11)) for j in axxrange])
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(formatter)
    if axyrange:
        major_locator = plt.FixedLocator(axyrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(1, 11)) for j in axyrange])
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_major_formatter(formatter)
