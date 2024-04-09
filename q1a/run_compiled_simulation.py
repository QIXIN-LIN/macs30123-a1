import numpy as np
import time
import scipy.stats as sts

# Import the compiled module
from health_index_simulation import simulate_health_index

start_time = time.time()
# Set the parameters and the random shocks as before
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters, draw all idiosyncratic random shocks,
# and create empty containers
S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulatio
np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

# Measure the performance with the AOT compiled function
z_mat = simulate_health_index(rho, mu, sigma, z_0, eps_mat)
elapsed_time = time.time() - start_time

print(f"Elapsed time with AOT-compiled code: {elapsed_time} seconds")
