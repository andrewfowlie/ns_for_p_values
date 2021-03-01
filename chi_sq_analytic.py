from scipy.stats import norm, chi2

from p_value import dynesty, brute, mn, Result

observed = 30.
n_dim = 5

def analytic_p_value(observed_, n_dim):
    p = chi2.sf(observed_, n_dim)
    return Result(p, 0, None)

def transform(cube):
    return norm.ppf(cube)
    
def test_statistic(observed_):
    return (observed_**2).sum()
    
print(analytic_p_value(observed, n_dim))
print(brute(test_statistic, transform, n_dim, observed, n=50000))
#print(dynesty(test_statistic, transform, n_dim, observed, n_live=100))
print(mn(test_statistic, transform, n_dim, observed, n_live=100))
