from scipy.stats import chi2, norm
from scipy.special import ndtri
import pickle
from matplotlib.ticker import FormatStrFormatter

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


if __name__ == "__main__":
    
    #print(analytic_p_value(observed, n_dim))
    #print(brute(test_statistic, transform, n_dim, observed, n=50000))
    #print(dynesty(test_statistic, transform, n_dim, observed, n_live=100))
    #print(mn(test_statistic, transform, n_dim, observed, n_live=100))
    #print(pc(test_statistic, transform, n_dim, observed, n_live=100))
              
    # Plot significance against calls
   
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.style.use("seaborn-colorblind")
    plt.grid(True)
    
    rel_error = 0.1         

    # PolyChord - expensive so save this data
    
    n_live = 100
    
    for d in [1, 2, 5, 10, 30]:

        pkl_name = "pc_dim_{}.pkl".format(d)
        
        try:
            with open(pkl_name, 'rb') as pkl:
                px, py = pickle.load(pkl)
        except:
            tmax = chi2.isf(norm.sf(7.), d)
            tmin = chi2.isf(norm.sf(0.), d)
            
            px = []
            py = []
                      
            for i, t in enumerate(np.geomspace(tmin, tmax, 20)):  
                       
                # Strategy is resume NS run, pushing threshold a bit further
                p = pc(test_statistic, transform, d, t, n_live=n_live, resume=i!=0)          
                true_ = analytic_p_value(t, d)
                
                ns_rel_error = (- np.log(true_.p_value) / n_live)**0.5
                scale = (ns_rel_error / rel_error)**2 
                
                # showing true significance here - could show calculated one
                px.append(true_.significance)
                py.append(p.calls * scale)
               
            with open(pkl_name, 'wb') as pkl: 
                pickle.dump((px, py), pkl)
            
        plt.plot(px, py, label="PolyChord. $d = {}$".format(d), ls="--")      
  
    # MultiNest - expensive so save this data
        
    # Reset colors so same as PolyChord
    plt.gca().set_prop_cycle(None)
    
    for d in [1, 2, 5, 10, 30]:

        pkl_name = "mn_dim_{}.pkl".format(d)
        
        try:
            with open(pkl_name, 'rb') as pkl:
                px, py = pickle.load(pkl)
        except:
            #continue
            tmax = chi2.isf(norm.sf(7.), d)
            tmin = chi2.isf(norm.sf(0.), d)
            
            px = []
            py = []
            
            # Cannot resume NS run so one long run
            p, ev_data = mn(test_statistic, transform, d, tmax, wrapped_params=[1] * d, n_live=n_live, sampling_efficiency=0.3, verbose=True, ev_data=True)   
              
            # extract number of calls
            threshold = np.sort(np.array(ev_data[-1]))

            for t in np.unique(threshold):
                           
                if t < tmin or t > tmax:
                    continue
                    
                calls = (threshold <= t).sum()
                true_ = analytic_p_value(t, d)
                
                ns_rel_error = (- np.log(true_.p_value) / n_live)**0.5
                scale = (ns_rel_error / rel_error)**2 
                
                # showing true significance here - could show calculated one
                px.append(true_.significance)
                py.append(calls * scale) 
            
            with open(pkl_name, 'wb') as pkl: 
                pickle.dump((px, py), pkl)
            
        plt.plot(px, py, label="MultiNest. $d = {}$".format(d), ls=":")   
         
    
    # MC and perfect NS
    
    d = 1  # independent of d in this setting
    tmax = chi2.isf(norm.sf(7.), d)
    tmin = chi2.isf(norm.sf(0.), d)
    x = []
    mc = []
    pns = []

    for t in np.geomspace(tmin, tmax, 500):        
        r = analytic_p_value(t, d)
        x.append(r.significance)
        
        mc.append(1. / (rel_error**2 * r.p_value))

        n_live = - np.log(r.p_value) / rel_error**2
        calls = -n_live * np.log(r.p_value)
        pns.append(calls)      
      
    plt.plot(x, mc, label="Monte Carlo", c="Crimson") 
    plt.plot(x, pns, label="Perfect NS", c="k")     
                        
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1d$\sigma$'))
    plt.legend(fontsize=9)
    plt.title("Computing $p$-value to 10% uncertainty")
    plt.xlabel("Significance, $Z$")
    plt.ylabel("Function calls (proxy for speed)")
    plt.gca().set_yscale('log')    
    plt.savefig("performance.pdf")
