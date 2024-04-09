from numba.pycc import CC
import numpy as np

cc = CC('health_index_simulation')

@cc.export('simulate_health_index', 'f8[:,:](f8, f8, f8, f8, f8[:,:])')
def simulate_health_index(rho, mu, sigma, z_0, eps_mat):
    S, T = eps_mat.shape
    z_mat = np.zeros((T, S))
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat


if __name__ == "__main__":
    cc.compile()