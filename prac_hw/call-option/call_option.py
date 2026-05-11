
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SEED = 777

# Assignment parameters
R = 0.05
SIGMA = 0.1
S0 = 100.0
K = 100.0
T = 1.0
N = 1_000_000


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(s0: float, k: float, r: float, sigma: float, tau: float) -> float:
    if tau <= 0.0:
        return max(s0 - k, 0.0)
    sig_sqrt_t = sigma * math.sqrt(tau)
    y_plus = (math.log(s0 / k) + tau * (r + 0.5 * sigma * sigma)) / sig_sqrt_t
    y_minus = (math.log(s0 / k) + tau * (r - 0.5 * sigma * sigma)) / sig_sqrt_t
    return s0 * norm_cdf(y_plus) - k * math.exp(-r * tau) * norm_cdf(y_minus)


def monte_carlo_call_iter(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    tau: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """S_i = S_{i-1} (1 + r*delta + sigma*sqrt(delta)*X_i)"""
    delta = tau / n_steps
    s = np.full(n_paths, s0, dtype=np.float64)
    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        s *= 1.0 + r * delta + sigma * math.sqrt(delta) * z
    payoff = np.maximum(s - k, 0.0)
    mean_pay = float(np.mean(payoff))
    std_pay = float(np.std(payoff, ddof=1))
    disc = math.exp(-r * tau)
    price = disc * mean_pay
    stderr_price = disc * std_pay / math.sqrt(n_paths)
    return price, stderr_price


def monte_carlo_call_exact(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    tau: float,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0,1)"""
    z = rng.standard_normal(n_paths)
    st = s0 * np.exp((r - 0.5 * sigma * sigma) * tau + sigma * math.sqrt(tau) * z)
    payoff = np.maximum(st - k, 0.0)
    mean_pay = float(np.mean(payoff))
    std_pay = float(np.std(payoff, ddof=1))
    disc = math.exp(-r * tau)
    price = disc * mean_pay
    stderr_price = disc * std_pay / math.sqrt(n_paths)
    return price, stderr_price


def ruN_experiment(rng: np.random.Generator) -> None:
    analytical = black_scholes_call(S0, K, R, SIGMA, T)
    print("Parameters:")
    print(f"r={R}, sigma={SIGMA}, S0={S0}, K={K}, T={T}, N={N}")
    print(f"Analytical price (Black-Scholes): {analytical:.6f}")

    t0 = time.perf_counter()
    mc_exact, se_exact = monte_carlo_call_exact(S0, K, R, SIGMA, T, N, rng)
    t1 = time.perf_counter()
    print("\nMonte Carlo (exact S_T sampling)")
    print(f"Estimate: {mc_exact:.6f}")
    print(f"Standard error (s/sqrt(N), discounted): {se_exact:.6f}")
    print(f"Deviation from analytical: {mc_exact - analytical:+.6f}")
    print(f"Elapsed: {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    n_steps = 100
    mc_euler, se_euler = monte_carlo_call_iter(S0, K, R, SIGMA, T, N, n_steps, rng)
    t1 = time.perf_counter()
    print(f"\nMonte Carlo (N={N}, n_steps={n_steps})")
    print(
        f"Estimate: {mc_euler:.6f} "
        f"(scheme bias O(delta); large n_steps and N approach BS)"
    )
    print(f"Standard error: {se_euler:.6f}")
    print(f"Deviation from analytical: {mc_euler - analytical:+.6f}")
    print(f"Elapsed: {t1 - t0:.3f} s")


def convergence_experiment(rng: np.random.Generator, out_dir: Path | str) -> None:
    analytical = black_scholes_call(S0, K, R, SIGMA, T)
    n_list = np.unique(np.logspace(2, 6, num=25, dtype=np.int64))  # ~100 to 1e6
    errors = []
    stderrs = []
    for n in n_list:
        mc, se = monte_carlo_call_exact(S0, K, R, SIGMA, T, int(n), rng)
        errors.append(abs(mc - analytical))
        stderrs.append(se)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(n_list, errors, "o-", label=r"$|\hat{C} - C_{\mathrm{BS}}|$", markersize=5)
    ax.loglog(n_list, stderrs, "s--", label="Sample SE of estimate", markersize=4)
    ref = stderrs[-1] * math.sqrt(float(n_list[-1])) / np.sqrt(n_list.astype(float))
    ax.loglog(n_list, ref, "k:", alpha=0.7, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel(r"$N$ (number of paths)")
    ax.set_ylabel("Error / SE")
    ax.set_title("Monte Carlo convergence to analytical price")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    path = Path(out_dir) / "convergence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Convergence plot saved: {path}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs("pics", exist_ok=True)
    ruN_experiment(rng)
    convergence_experiment(rng, Path(__file__).parent)


if __name__ == "__main__":
    main()
