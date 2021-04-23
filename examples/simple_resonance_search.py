import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from scipy.special import logsumexp, ndtri
from tqdm import tqdm

sys.path.append("./..")
sys.path.append("./../p_value")
from p_value.ns import pc, ns_result
from p_value.brute import brute_low_memory


def very_simple_ts(data):
    return (data**2).max()

def very_simple_transform(u):
    return ndtri(u)

def three_bin_ts(data):
    a, b, c = data[0:-2], data[1:-1], data[2:]
    dt = 0.125 * (a**2 + 6. * a * b + 5. * b**2 + 2. * a * c + 6. * b * c + c**2)
    return dt.max()

def three_bin_ts_loop(data):
    n_bins = len(data)
    ts_0 = np.array([sum(data[cbin-1:cbin+2]**2) for cbin in range(1,n_bins-1)])
    ts_1 = []
    for cbin in range(1,n_bins-1):
        lam = 0.5*sum(data[cbin-1:cbin+2])
        ts_1.append((0.5*lam-data[cbin-1])**2 + (lam-data[cbin])**2 + (0.5*lam-data[cbin+1])**2)
    ts_1 = np.array(ts_1)
    return np.max(ts_0-ts_1)

def get_line(file, label):
    with open(file) as res_file:
        for line in res_file:
            if line.strip() == label:
                next_line = res_file.readline()
                return np.array([float(e) for e in next_line.split()])
        else:
            raise RuntimeError("didn't find {}".format(label))

def analyse_pch(root="pc_simple", n_live=100):
    # Analyse PCh results
    n_live = n_live
    root = root
    label1 = "=== local volume -- log(<X_p>) ==="
    label2 = "=== Number of likelihood calls ==="

    log_xp = get_line("chains/"+root+".resume", label1)
    log_x = logsumexp(log_xp)
    n_iter = -log_x * n_live
    print(log_x / np.log(10.))
    # get ev data
    ev_name = "chains/"+root+"_dead.txt"
    ev_data = np.genfromtxt(ev_name)
    test_statistic = ev_data[:, 0]
    log_x = -np.arange(0, len(test_statistic), 1.) / n_live
    log_x_delta = np.sqrt(-log_x / n_live)

    pch_calls = int(get_line("chains/"+root+".resume", label2)[0])
    
    res = ns_result(n_iter, n_live, pch_calls)
    
    return res, test_statistic, log_x

observed = 30.
ts_vals_range = np.linspace(0, observed, 200)

def calc_pvals(ts_vals):
    ts_sorted = np.sort(ts_vals)
    n_ts = float(len(ts_sorted))
    pvals_range = []
    i = 1
    for ts in np.flip(ts_vals_range):
        while (ts_sorted[-i] > ts) and (i < len(ts_sorted)):
            i += 1
        pvals_range.append((i-1)/n_ts)
    pvals_range = np.flip(pvals_range)
    pval_errs_range = np.array([np.sqrt(p/n_ts) for p in pvals_range])
    return pvals_range, pval_errs_range

if __name__ == "__main__":

    # Brute MC
    n_samples = int(1e6)
    #ts_vals = [very_simple_ts(norm.rvs(loc=0, scale=1, size=30)) for i in range(n_samples)]
    #ts_vals_threebin = [three_bin_ts(norm.rvs(loc=0, scale=1, size=30)) for i in tqdm(range(n_samples))]

    b = brute_low_memory(very_simple_ts, very_simple_transform, 30, ts_vals_range, n=n_samples)
    b_3b = brute_low_memory(three_bin_ts, very_simple_transform, 30, ts_vals_range, n=n_samples)

    # Polychord
    res1, res2 = pc(very_simple_ts, very_simple_transform, n_dim=30, observed=observed, n_live=100, file_root="pc_simple", do_clustering=False, feedback=2, resume=False, ev_data=True)
    res1_3b, res2_3b = pc(three_bin_ts, very_simple_transform, n_dim=30, observed=observed, n_live=100, file_root="pc_3bin", do_clustering=False, feedback=2, resume=False, ev_data=True)

    #p_vals_range, _ = calc_pvals(ts_vals)
    #p_vals_threebin_range, _ = calc_pvals(ts_vals_threebin)
    res, test_statistic, log_x = analyse_pch()
    res_3b, test_statistic_3b, log_x_3b = analyse_pch("pc_3bin")

    # analytic in this case
    log10_local_p = chi2.logsf(ts_vals_range, df=1) / np.log(10)
    log10_global_p = np.log10(30.) + log10_local_p

    plt.plot(ts_vals_range, log10_global_p, c='red', ls='--', label="Theory")
    plt.plot(ts_vals_range, np.log10(b), c='grey', ls='--', label="Brute MC")
    plt.plot(test_statistic, np.log10(np.exp(log_x)), c='b', label="Polychord")
    plt.xlim([0,observed])
    plt.xlabel('TS')
    plt.ylabel('$\log_{10}(p)$')
    plt.legend(title='1-bin example')
    plt.savefig("simple_ts_onebin.pdf")
    plt.show()

    plt.plot(ts_vals_range, np.log10(b_3b), c='grey', ls='--', label="Brute MC")
    plt.plot(test_statistic_3b, np.log10(np.exp(log_x_3b)), c='b', label="Polychord")
    plt.xlim([0,observed])
    plt.xlabel('TS')
    plt.ylabel('$\log_{10}(p)$')
    plt.legend(title='3-bin example')
    plt.savefig("simple_ts_threebin.pdf")
    plt.show()
    

