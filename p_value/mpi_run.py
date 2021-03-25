import sys
import time
from datetime import datetime

import numpy as np
from scipy.stats import poisson
from mpi4py import MPI

sys.path.insert(0, '../examples/')
import higgs_functions as higgs

n_batch_size = 250
n_batches = 400
n_tasks = n_batch_size*n_batches

wrapper_mapping = { 'higgs': higgs.calculate_ts }

output_path = sys.argv[1]
try:
    wrapper_fun = wrapper_mapping[sys.argv[2]]
except KeyError:
    raise ValueError('Invalid wrapper selected!')

def current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_process_batch(index):
    res = np.array([wrapper_fun(rvs) for rvs in poisson.rvs(n_batch_size*[higgs.expected_bkg])])
    out_file_name = output_path+'/temp/tsval_batch_{:d}.dat'.format(int(index))
    np.savetxt(out_file_name, res.T, fmt='%.6e')
    tsvals = np.array(res[:,-1])
    return tsvals


# Set up the MPI environment and variables.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ncores = comm.Get_size()
start_time = time.time()

# Let the workers calculate the flux.
if (rank > 0):
    tsvals = run_process_batch(rank-1)
    comm.Send(tsvals, dest=0)
    for i in range(n_batches):
        task_id = comm.recv(source=0)
        if (task_id > n_batches):
            break
        index = int(task_id-1)
        tsvals = run_process_batch(index)
        comm.Send(tsvals, dest=0)
    print('{}: MPI rank {} finished! MC simulations took {:.1f} mins.'.format(current_datetime(), rank, (time.time()-start_time)/60.0))

# Let the master process distribute tasks and receive results:
if (rank == 0):
    print('Master process waiting for {} results from {} other processes...'.format(n_tasks, ncores-1), flush=True)
    all_results = np.array([])
    for task_id in range(ncores, n_batches+ncores):
        info = MPI.Status()
        buf = np.zeros(n_batch_size, dtype='float64')
        comm.Recv(buf, source=MPI.ANY_SOURCE, status=info)
        all_results = np.concatenate((all_results,buf))
        worker_id = info.Get_source()
        comm.send(task_id, dest=worker_id)
        if ((task_id % 50 == 0)|(task_id==n_batches+ncores-1)):
            print('Calculated another 50 batches of size {}. Currently at: {}'.format(task_id, n_batch_size), flush=True)
    print('{}: All MPI tasks finished after {:.1f} mins!'.format(current_datetime(), rank, (time.time()-start_time)/60.0), flush=True)
    out_file_name = output_path+'/all_tsvals.dat'
    print('Formatting results and saving them to '+out_file_name+'.', flush=True)
    np.savetxt(out_file_name, all_results.T, fmt='%.6e')
    print('All tasks complete! Finishing MPI routine now...', flush=True)
    MPI.Finalize()
