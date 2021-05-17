"""
Obhect for stroing and showing p-value result
=============================================
"""

import warnings
from scipy.stats import norm
import numpy as np


class Result(object):
    """
    Result of p-value computation
    """
    def __init__(self, p_value, p_value_uncertainty, calls):
        self.p_value = p_value
        self.p_value_uncertainty = p_value_uncertainty
        self.calls = calls

    @property
    def log10_pvalue(self):
        return np.log10(self.p_value)

    @property
    def error_log10_pvalue(self):
        try:
            error_log_pvalue = self.p_value_uncertainty / self.p_value
            return error_log_pvalue / np.log(10.)
        except ZeroDivisionError:
            return np.inf

    @property
    def significance(self):
        return norm.isf(self.p_value)

    def __str__(self):
        pval_exponent = -np.inf
        if self.p_value > 0:
            pval_exponent = int(np.floor(np.log10(self.p_value)))
            return "p-value = ({:.4f} +/- {:.4f})e{:03d}. log10(p-value) = {:.5} +/- {:.5}. Signifiance = {:.3f} sigma. Function calls = {}".format(
                self.p_value/10**pval_exponent, self.p_value_uncertainty/10**pval_exponent, pval_exponent,
                self.log10_pvalue, self.error_log10_pvalue,
                self.significance, self.calls)

        warnings.warn("p-value < 0. %s", self.p_value)
        return "p-value = {}".format(self.p_value)

    @classmethod
    def from_ns(cls, n_iter, n_live, calls):
        """
        Constructor for NS result.
        """
        log_x = - float(n_iter) / n_live
        p_value = np.exp(log_x)
        log_x_uncertainty = (-log_x / n_live)**0.5
        p_value_uncertainty = p_value * log_x_uncertainty
        return cls(p_value, p_value_uncertainty, calls)
