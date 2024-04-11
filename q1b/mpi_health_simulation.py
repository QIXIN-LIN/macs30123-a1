from mpi4py import MPI
import numpy as np
import scipy.stats as sts
from health_index_simulation import simulate_health_index
import time

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters and draw all idiosyncratic random shocks
S = 1000 // size  # Divide the work among processes
T = int(4160)  # Set the number of periods for each simulation

# Ensure each core gets different random numbers
np.random.seed(rank)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))

# Create empty containers
z_mat = np.zeros((T, S))

# Start timing
start_time = time.time()

# Run the compiled simulation
z_mat = simulate_health_index(rho, mu, sigma, z_0, eps_mat)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Gather all timing data at the root process
all_times = comm.gather(elapsed_time, root=0)

if rank == 0:
    # Only the root process will output the data
    for i, time in enumerate(all_times):
        print(f"Core {i}: Simulation took {time} seconds.")