
import numpy as np
import matplotlib.pyplot as plt

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
    
    prac_mean = np.mean(B, axis=0)
    prac_var = np.var(B, axis=0)
    
    print(f"Results at T={T}:")
    print(f"Mean: {prac_mean[-1]:.4f}")
    print(f"Variance: {prac_var[-1]:.4f}")
    print(f"Theoretical Variance (Var=t): {T}")

    graph(t, B, prac_var)
    print(f"Builded {N_PATHS} paths with {n_steps} steps")

def graph(t, B, var):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap('viridis', N_PATHS)
    for i in range(N_PATHS): 
        ax1.plot(t, B[i], color=cmap(i), alpha=0.1, linewidth=0.5)
    
    ax1.set_title(f'Wiener Process: {N_PATHS} Paths')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('B(t)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, var, label='Practical Variance', color='blue', linewidth=2)
    ax2.plot(t, t, 'r--', label='Theoretical Variance (D=t)', linewidth=2)
    
    ax2.set_title('Variance')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wiener.png', dpi=150)
    plt.show()
    
    
    
if __name__ == "__main__":
    main()