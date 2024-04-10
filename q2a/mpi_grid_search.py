from mpi4py import MPI
import numpy as np
import scipy.stats as sts
from health_index_sim import simulate_individual_health_index

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

    # Initialize an array to store the average periods for each rho in the subrange
    avg_periods_subrange = np.zeros(len(rho_subrange))

    # Start timing the grid search
    start_time = MPI.Wtime()

    # Perform the grid search
    for i, rho in enumerate(rho_subrange):
        avg_periods = 0
        for s in range(S):
            periods = simulate_individual_health_index(rho, mu, sigma, z_0, eps_mat[:, s])
            avg_periods += periods
        avg_periods_subrange[i] = avg_periods / S

    # Gather all avg_periods data at the root process
    all_avg_periods = None
    if rank == 0:
        all_avg_periods = np.zeros(num_rho_values)
    comm.Gather(avg_periods_subrange, all_avg_periods, root=0)

    # End timing the grid search
    end_time = MPI.Wtime()

    # Compute the total time taken for the grid search
    total_time = end_time - start_time

    if rank == 0:
        # Find the optimal rho and its corresponding average number of periods
        optimal_idx = np.argmax(all_avg_periods)
        optimal_rho = rho_values[optimal_idx]
        max_avg_periods = all_avg_periods[optimal_idx]

        # Report the optimal rho, average number of periods, and total time
        print(f"Optimal ρ: {optimal_rho}, Average Number of Periods: {max_avg_periods}")
        print(f"Total computation time: {total_time} seconds")

        # Save the results to a file
        np.savetxt("rho_vs_periods.csv", np.column_stack((rho_values, all_avg_periods)), delimiter=",", header="rho,avg_periods", comments="")

if __name__ == "__main__":
    main()