
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42

T = 1.0
DELTA = 0.0001
N_PATHS = 1000

def main():
    np.random.seed(SEED)
    
    n_steps = int(T / DELTA)
    t = np.linspace(0, T, n_steps + 1)

    xi = np.random.randn(N_PATHS, n_steps)
    B = np.cumsum(np.sqrt(DELTA) * xi, axis=1)
    B = np.insert(B, 0, 0, axis=1)  # B_0 = 0

    graph(t, B)
    print(f"Builded {N_PATHS} paths with {n_steps} steps")

def graph(t, B):
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('viridis', N_PATHS)
    for i in range(N_PATHS):
        plt.plot(t, B[i], color=cmap(i), alpha=0.1, linewidth=0.5)

    plt.title(f'Wiener process: {N_PATHS} paths\nT={T}, δ={DELTA}', fontsize=14)
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('B(t)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('wiener_1000_paths.png', dpi=150)
    plt.show()
    
    
    
if __name__ == "__main__":
    main()