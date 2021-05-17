import sys
import os
import time
from datetime import datetime

import numpy as np
from scipy.stats import poisson
from mpi4py import MPI

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import higgs_functions as higgs

wrapper_mapping = { 'higgs': higgs.nested_ts, 'higgs_full': higgs.nested_ts_full, 'higgs_af_fxbkg': af.nested_ts_fixed_bkg, 'higgs_fxbkg': higgs.nested_ts_bkg, 'higgs_sbkg': higgs.nested_ts_simple, 'higgs_fast': higgs.nested_ts_simple_fast }

output_path = sys.argv[1]
try:
    wrapper_fun = wrapper_mapping[sys.argv[2]]
except KeyError:
    raise ValueError('Invalid wrapper selected!')

n_batch_size = int(sys.argv[3])
n_batches = int(sys.argv[4])
five_percent_batch = int(0.05*n_batches)
n_tasks = n_batch_size*n_batches

def current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_process_batch(index):
    res0 = np.array([wrapper_fun(data) for data in [poisson.rvs(higgs.expected_bkg) for i in range(n_batch_size)]])
    # res0 = wrapper_fun(index, n_batch_size)
    #out_file_name = output_path+'/temp/tsval_{}_batch_{:d}.dat'.format(sys.argv[2],int(index))
    #np.savetxt(out_file_name, res0, fmt='%.6e')
    #res1 = np.array(res0[:,-1])
    res1 = np.array(res0)
    return res1


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
        if ((task_id % five_percent_batch == 0)|(task_id==n_batches+ncores-1)):
            print('Calculated another {} batches of size {}. Currently at task id {}; {}% complete'.format(five_percent_batch, n_batch_size, task_id, int(100*task_id/n_batches)), flush=True)
            out_file_name = output_path+'/temp/tsval_{}_batch_{:d}.dat'.format(sys.argv[2],int(task_id/five_percent_batch))
            np.savetxt(out_file_name, all_results[-five_percent_batch*n_batch_size:].T, fmt='%.6e')
    print('{}: All MPI tasks finished after {:.1f} mins!'.format(current_datetime(), rank, (time.time()-start_time)/60.0), flush=True)
    out_file_name = output_path+'/run5_tsvals_'+sys.argv[2]+'.dat'
    print('Formatting results and saving them to '+out_file_name+'.', flush=True)
    np.savetxt(out_file_name, all_results.T, fmt='%.6e')
    print('All tasks complete! Finishing MPI routine now...', flush=True)
    MPI.Finalize()
