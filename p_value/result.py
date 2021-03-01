"""
P-value computation elements
"""

from scipy.stats import norm

class Result(object):
    """
    Result of p-value computation
    """
    def __init__(self, p_value, p_value_uncertainty, calls):
        self.p_value = p_value
        self.p_value_uncertainty = p_value_uncertainty
        self.calls = calls

    @property
    def z(self):
        return norm.isf(self.p_value)

    def __str__(self):
        return "p-value = {} +/- {}. Signifiance = {} sigma. Function calls = {}".format(
                self.p_value, self.p_value_uncertainty, self.z, self.calls)
