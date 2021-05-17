"""
Unit-test p-value calculations
==============================
"""

import unittest

from p_value import mn, pc
from chi_sq_analytic import analytic_p_value, transform, test_statistic


class TestNestedSampling(unittest.TestCase):

    def test_mn_d_4(self):
        d = 4
        observed = 16.25

        a = analytic_p_value(observed, d)
        ns = mn(test_statistic, transform, d, observed, n_live=100, seed=87, sampling_efficiency=0.3)

        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def test_pc_d_4(self):
        d = 4
        observed = 16.25

        a = analytic_p_value(observed, d)
        ns = pc(test_statistic, transform, d, observed, n_live=100, seed=87)

        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def test_mn_d_1(self):
        d = 1
        observed = 4.

        a = analytic_p_value(observed, d)
        ns = mn(test_statistic, transform, d, observed, n_live=100, seed=87, sampling_efficiency=0.3)
        print(ns, a)
        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def test_pc_d_1(self):
        d = 1
        observed = 4.

        a = analytic_p_value(observed, d)
        ns = pc(test_statistic, transform, d, observed, n_live=100, num_repeats=3, seed=87)

        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

if __name__ == '__main__':
    unittest.main(verbosity=10)
