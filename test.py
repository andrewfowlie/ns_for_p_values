from p_value import mn, pc, brute
from chi_sq_analytic import analytic_p_value, transform, test_statistic

import unittest


class TestNestedSampling(unittest.TestCase):

    def atest_ns_d_4(self):
        d = 4
        observed = 16.25
        
        a = analytic_p_value(observed, d)
        ns = mn(test_statistic, transform, d, observed, n_live=1000, seed=87, sampling_efficiency=0.3)
    
        print(a, ns)
        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def atest_pc_d_4(self):
        d = 4
        observed = 16.25
        
        a = analytic_p_value(observed, d)
        ns = pc(test_statistic, transform, d, observed, n_live=1000, seed=87)
    
        print(a, ns)
        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def atest_ns_d_1(self):
        d = 1
        observed = 4.
        
        a = analytic_p_value(observed, d)
        ns = mn(test_statistic, transform, d, observed, n_live=4000, seed=87, sampling_efficiency=0.3)
    
        print(a, ns)
        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

    def test_pc_d_1(self):
        d = 1
        observed = 4.
        
        a = analytic_p_value(observed, d)
        ns = pc(test_statistic, transform, d, observed, n_live=4000, num_repeats=10)
    
        print(a, ns)
        self.assertAlmostEqual(ns.log10_pvalue, a.log10_pvalue, delta=3. * ns.error_log10_pvalue)

if __name__ == '__main__':
    unittest.main(verbosity=10)
