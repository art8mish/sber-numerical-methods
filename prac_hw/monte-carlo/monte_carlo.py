import numpy as np
import time
import matplotlib.pyplot as plt

PI = np.pi
SEED = 777

def monte_carlo_pi(n_points):
    x = np.random.random(n_points)
    y = np.random.random(n_points)
    inside = (x*x + y*y) <= 1.0
    hits = np.sum(inside)
    return 4.0 * hits / n_points

    
    
def main():
    np.random.seed(SEED)
    n = 100000000
    start = time.time()
    pi_est = monte_carlo_pi(n)
    elapsed = time.time() - start
    print(f"Theoretical π = {PI:.6f}")
    print(f"Practical π = {pi_est:.6f}")
    print(f"Elapsed time: {elapsed:.2f}s")
    
    standart_error()
    

def th_error(N):
    return np.sqrt(PI * (4 - PI) / N)

def standart_error():
    N_values = np.logspace(1, 8, 50, dtype=int)
    se_th = th_error(N_values)
    num_trials = 100
    max_total_points = 100000000
    empirical_N = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    empirical_se = []

    for N in empirical_N:
        pi_estimates = []
        trials = min(num_trials, max_total_points // N)
        for _ in range(trials):
            pi = monte_carlo_pi(N)
            pi_estimates.append(pi)
        empirical_se.append(np.std(pi_estimates, ddof=1))

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, se_th, 'b-', label='Theoretical SE', linewidth=2)
    plt.loglog(empirical_N, empirical_se, 'ro', markersize=8, label='Practical SE')
    plt.xlabel('N')
    plt.ylabel('Standart error')
    plt.title('Standart error')
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('standard_error.png', dpi=150)
    plt.show()
    

if __name__ == "__main__":
    main()
