from mpi4py import MPI
import numpy as np
import scipy.stats as sts
from health_index_sim import simulate_individual_health_index  # Import the pre-compiled function

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the model parameters
    mu = 3.0
    sigma = 1.0
    z_0 = mu - 3 * sigma  # Starting health index below average
    T = 4160  # Number of periods
    S = 1000  # Number of lives to simulate
    num_rho_values = 200  # Number of rho values to test

    # Generate the matrix of shocks only once on rank 0
    if rank == 0:
        np.random.seed(0)
        eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    else:
        eps_mat = None

    # Broadcast the shock matrix to all processes
    eps_mat = comm.bcast(eps_mat, root=0)

    # Divide the range of rho values among processes
    rho_values = np.linspace(-0.95, 0.95, num_rho_values)
    rho_subrange = np.array_split(rho_values, size)[rank]

    # Prepare to gather results
    max_periods_array = np.zeros(1)
    optimal_rho_array = np.zeros(1)

    # Start the timer
    start_time = MPI.Wtime()

    # Perform the grid search
    max_periods = -1
    optimal_rho = None
    for rho in rho_subrange:
        # Simulate the health index for each individual and each period
        avg_periods = 0
        for s in range(S):
            periods = simulate_individual_health_index(rho, z_0, T, 1, eps_mat[:, s])
            avg_periods += periods
        avg_periods /= S
        # Check if this rho value gives a longer time before health index goes negative
        if avg_periods > max_periods:
            max_periods = avg_periods
            optimal_rho = rho

    # Store the results
    max_periods_array[0] = max_periods
    optimal_rho_array[0] = optimal_rho if optimal_rho is not None else -1  # Use -1 to indicate no rho found

    # Reduce the results to the root process
    global_max_periods = np.zeros(1)
    global_optimal_rho = np.zeros(1)
    comm.Reduce(max_periods_array, global_max_periods, op=MPI.MAX, root=0)
    comm.Reduce(optimal_rho_array, global_optimal_rho, op=MPI.MAX, root=0)

    # Stop the timer
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Optimal rho: {global_optimal_rho[0]}, which gives an average of {global_max_periods[0]} periods before health index goes negative.")
        print(f"Total computation time: {end_time - start_time} seconds.")

if __name__ == "__main__":
    main()
