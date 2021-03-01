# p_value module

Contains methods for 

- p-value from brute force
- p-value from MultiNest nested sampling
- p-value from dynesty nested sampling

I hacked the usual APIs to force nested sampling to terminate at the appropriate point, and to 
extract a p-value estimate from the result. MultiNest though doens't always seem to run for long enough before
some other convergence criteria are met.

# example

chi_sq_analytic.py uses the above methods to compute the p-value. The observations are `n_dim` standard normals. The test-statistic is the sum of them squared, whch should
follow a chi-sq distribution with `n_dim` dof. 

Brute force and NS both appear to work, NS being more efficient. Though dynesty doesn't work nearly as well as MulitNest at the moment.


