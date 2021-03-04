"""
P-value computation elements
"""

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
        return "p-value = {} +/- {}. log10 p-value = {} +/- {}. Signifiance = {} sigma. Function calls = {}".format(
            self.p_value, self.p_value_uncertainty,
            self.log10_pvalue, self.error_log10_pvalue,
            self.significance, self.calls)
