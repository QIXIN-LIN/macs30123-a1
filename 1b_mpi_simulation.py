from mpi4py import MPI
from numba import jit
import numpy as np
import scipy.stats as sts
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Simulation parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu
T = 4160
S_total = 1000  # Total number of simulations, change for test

# Divide the simulations among available cores
S_per_core = S_total // size
remaining = S_total % size

if rank < remaining:
    S_per_core += 1

# Each process generates its portion of eps_mat with a unique seed
np.random.seed(rank)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S_per_core))

@jit(nopython=True)
def simulate_health_index(eps_mat, S, T, rho, mu, z_0):
    z_mat = np.zeros((T, S), dtype=np.float64)
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat

start_time = time.time()
z_mat = simulate_health_index(eps_mat, S_per_core, T, rho, mu, z_0)
end_time = time.time()

# Measure time and report back to the master process
execution_time = end_time - start_time
execution_times = comm.gather(execution_time, root=0)

if rank == 0:
    # Process and display the gathered execution times
    print(f"Execution times: {execution_times}")