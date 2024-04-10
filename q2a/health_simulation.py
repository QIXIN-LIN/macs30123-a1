from numba.pycc import CC

cc = CC('health_index_sim')

@cc.export('simulate_individual_health_index', 'i8(f8, f8, f8, f8, f8[:])')
def simulate_individual_health_index(rho, mu, sigma, z_0, epsilons):
    T = len(epsilons)
    zt = z_0
    for t in range(T):
        zt = rho * zt + (1 - rho) * mu + epsilons[t]
        if zt <= 0:
            return t  # Return the time period when zt falls to or below zero
    return T  # If zt never falls to or below zero, return the last time period

if __name__ == "__main__":
    cc.compile()