from scipy.stats import chi2
from scipy.special import ndtri

from p_value import dynesty, brute, mn, pc, Result

observed = 110.
n_dim = 30

def analytic_p_value(observed_, n_dim):
    p = chi2.sf(observed_, n_dim)
    return Result(p, 0, None)

def transform(cube):
    return ndtri(cube)
    
def test_statistic(observed_):
    return (observed_**2).sum()
    
print(analytic_p_value(observed, n_dim))
print(brute(test_statistic, transform, n_dim, observed, n=50000))
#print(dynesty(test_statistic, transform, n_dim, observed, n_live=100))
#print(mn(test_statistic, transform, n_dim, observed, n_live=100))
print(pc(test_statistic, transform, n_dim, observed, n_live=100))
