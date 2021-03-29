import sys
import os
from tqdm import tqdm
import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir+'/../examples/')
from nested_errors import get_mn_pval_single


results = []
for i in tqdm(range(1000)):
    a = get_mn_pval_single(i, file_root="./temp/mn", n_dim=4, target_obs=9.0)
    results.append(a)

print(np.array(results))
