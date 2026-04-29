import os
import numpy as np
import time
import matplotlib.pyplot as plt

PI = np.pi
SEED = 777


def box_muller_standard_normals(n):
    m = (n + 1) // 2
    u1 = np.random.random(m)
    u2 = np.random.random(m)
    u1 = np.maximum(u1, 1e-300)
    r = np.sqrt(-2.0 * np.log(u1))
    t = 2.0 * PI * u2
    z1 = r * np.cos(t)
    z2 = r * np.sin(t)
    z = np.empty(2 * m, dtype=np.float64)
    z[0::2] = z1
    z[1::2] = z2
    return z[:n]


def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * PI))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def lognormal_pdf(x, mu, sigma):
    x = np.maximum(x, 1e-300)
    return (1.0 / (x * sigma * np.sqrt(2.0 * PI))) * np.exp(
        -0.5 * ((np.log(x) - mu) / sigma) ** 2
    )


def demo_normal_lognormal(n_sample=100_000, mu=0.5, sigma=0.5):
    z = box_muller_standard_normals(n_sample)
    g = mu + sigma * z
    L = np.exp(g)
    log_L = np.log(L)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    bins_g = 60
    axes[0].hist(g, bins=bins_g, density=True, alpha=0.65, color="steelblue", label="sample")
    gx = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    axes[0].plot(gx, normal_pdf(gx, mu, sigma), "r-", lw=2, label=r"$N(\mu,\sigma^2)$")
    axes[0].set_title(r"Нормальное $G$")
    axes[0].set_xlabel(r"$g$")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    L_hi = max(np.percentile(L, 99.5) * 1.05, np.exp(mu + 3 * sigma))
    Lx = np.linspace(1e-6, L_hi, 500)
    axes[1].hist(L, bins=bins_g, density=True, alpha=0.65, color="seagreen", label="sample")
    axes[1].set_xlim(0, L_hi)
    axes[1].plot(Lx, lognormal_pdf(Lx, mu, sigma), "r-", lw=2, label="lognormal PDF")
    axes[1].set_title(r"Логнормальное $L=e^G$")
    axes[1].set_xlabel(r"$L$")
    axes[1].set_ylabel("density")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.5)

    axes[2].hist(log_L, bins=bins_g, density=True, alpha=0.65, color="coral", label=r"$\ln L$")
    axes[2].plot(gx, normal_pdf(gx, mu, sigma), "r-", lw=2, label=r"$N(\mu,\sigma^2)$")
    axes[2].set_title(r"$\ln L$ совпадает с $G$")
    axes[2].set_xlabel(r"$\ln L$")
    axes[2].set_ylabel("density")
    axes[2].legend()
    axes[2].grid(True, ls="--", alpha=0.5)

    fig.suptitle(
        rf"Box-Muller $\to$ $G\sim N(\mu,\sigma^2)$, $L=\exp(G)$ ($\mu={mu}$, $\sigma={sigma}$, $n={n_sample}$)",
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs("pics", exist_ok=True)
    out = os.path.join("pics", "normal_lognormal.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()

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
    demo_normal_lognormal()

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
    os.makedirs("pics", exist_ok=True)
    plt.savefig(os.path.join("pics", "standard_error.png"), dpi=150)
    plt.show()
    

if __name__ == "__main__":
    main()
