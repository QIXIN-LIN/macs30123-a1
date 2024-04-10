# mpi_grid_search.py

from mpi4py import MPI
import numpy as np
import scipy.stats as sts
from health_index_sim import simulate_individual_health_index  # make sure this is the right path

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Model parameters
    mu = 3.0
    sigma = 1.0
    z_0 = mu - 3 * sigma
    T = 4160
    S = 1000
    num_rho_values = 200

    # Generate the shocks matrix on rank 0
    if rank == 0:
        np.random.seed(0)
        eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    else:
        eps_mat = None
    eps_mat = comm.bcast(eps_mat, root=0)

    # Distribute rho values across processes
    rho_values = np.linspace(-0.95, 0.95, num_rho_values)
    rho_subrange = np.array_split(rho_values, size)[rank]

    # Initialize local optimum
    local_optimal_rho = -2.0  # Start with an invalid rho value
    local_max_periods = -1.0

    for rho in rho_subrange:
        avg_periods = np.mean([simulate_individual_health_index(rho, z_0, T, 1, eps) for eps in eps_mat.T])
        if avg_periods > local_max_periods:
            local_max_periods = avg_periods
            local_optimal_rho = rho

    # Print the local optimum for each process
    print(f"rank = {rank}, opt_rho = {local_optimal_rho}")

    # Gather the results from all processes to the root process
    global_max_periods = comm.reduce(local_max_periods, op=MPI.MAX, root=0)
    global_optimal_rho = comm.reduce(local_optimal_rho, op=MPI.MAX, root=0)

    # The root process prints the global optimum and computation time
    if rank == 0:
        # Assuming the computation time is being measured correctly
        total_time = MPI.Wtime() - start_time
        print(f"rank = {rank}, total_time = {total_time}")
        print(f"rank = {rank}, opt_rho_index = {np.argmax(global_max_periods)}")  # This needs to be calculated correctly
        print(f"rank = {rank}, opt_rho_val = {global_optimal_rho}")

if __name__ == "__main__":
    start_time = MPI.Wtime()
    main()
