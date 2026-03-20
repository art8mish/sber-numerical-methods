import numpy as np
from scipy import stats

SEED = 777

def ks_test_normal(data):
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(0, 1))
    return ks_p

def two_level_test(generator, dist_test_func, n_samples=100, sample_size=1000, **kwargs):
    p_values = []
    for _ in range(n_samples):
        data = generator(sample_size)
        p = dist_test_func(data, **kwargs)
        p_values.append(p)
        
    ks_stat, ks_p = stats.kstest(p_values, 'uniform')
    return ks_p, p_values

def normal_generator(size):
    return np.random.normal(0, 1, size)

def generate_bits(n_bits):
    return np.random.randint(0, 2, n_bits)

def fourier_test(bits):
    n = len(bits)
    h = np.sqrt(2.9957 * n)
    s = 2 * bits - 1 # to +1/-1
    f = np.fft.fft(s)
    magnitude = np.abs(f)
    count = np.sum(magnitude < h)
    expected = 0.95 * n
    variance = 0.05 * 0.95 * n
    z = (count - expected) / np.sqrt(variance)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return p

def autocorrelation_test(bits, lag=1):
    n = len(bits)
    if lag >= n:
        return 1.0
    xor_sum = np.sum(bits[:n-lag] ^ bits[lag:])
    p = 2 * min(stats.binom.cdf(xor_sum, n-lag, 0.5),
                1 - stats.binom.cdf(xor_sum - 1, n-lag, 0.5))
    return p

def gaps_test(bits):
    ones = np.where(bits == 1)[0]
    if len(ones) < 2:
        return 1.0
    gaps = np.diff(ones) - 1 # num of zeros between consecutive ones
    # bins: 0,1,2,3,4,5 and more
    bins = [0, 1, 2, 3, 4, 5, np.inf]
    observed, _ = np.histogram(gaps, bins=bins)
    # theoretical: p=0.5
    probs = [0.5**(k+1) for k in range(5)]
    probs.append(1 - sum(probs)) # tail
    expected = np.array(probs) * len(gaps)
    chi2 = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)
    return p_value



def main():
    print("1) Normality test")
    ks_p_norm, _ = two_level_test(normal_generator, ks_test_normal,
                                n_samples=100, sample_size=1000)
    print(f"Two-level p-value: {ks_p_norm:.4f}")
    res = "passed" if ks_p_norm > 0.05 else "failed"
    print(f"Test {res}")

    print("2) Bit sequence tests")
    ks_p_fourier, _ = two_level_test(generate_bits, fourier_test,
                                      n_samples=50, sample_size=1024)
    print(f"Fourier test, two-level p-value: {ks_p_fourier:.4f}")

    ks_p_auto, _ = two_level_test(generate_bits, autocorrelation_test,
                                   n_samples=50, sample_size=10000)
    print(f"Autocorrelation (lag=1), two-level p-value: {ks_p_auto:.4f}")

    ks_p_gaps, _ = two_level_test(generate_bits, gaps_test,
                                   n_samples=50, sample_size=10000)
    print(f"Gaps test, two-level p-value: {ks_p_gaps:.4f}")

if __name__ == "__main__":
    np.random.seed(777)
    main()

    