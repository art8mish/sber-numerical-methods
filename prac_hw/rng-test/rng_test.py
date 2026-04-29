import numpy as np
from scipy import stats

SEED = 777
rng_norm = np.random.default_rng(SEED)
rng_bits = np.random.Generator(np.random.MT19937(SEED))

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
    return rng_norm.normal(0, 1, size)

def generate_bits(n_bits):
    n_words = (n_bits + 31) // 32
    words = rng_bits.bit_generator.random_raw(n_words).astype(np.uint32)
    bits = np.unpackbits(words.view(np.uint8), bitorder='little')
    return bits[:n_bits]

def bad_generate_bits(n_bits):
    bad = generate_bits(n_bits).copy()
    bad[::32] = 0
    return bad

def fourier_test(bits):
    n = len(bits)
    h = np.sqrt(2.9957 * n)
    s = 2 * bits.astype(np.int8) - 1  # to +1/-1
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
    gaps = np.diff(ones) - 1
    bins = [0, 1, 2, 3, 4, 5, np.inf]
    observed, _ = np.histogram(gaps, bins=bins)
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
    print(f"Good RNG - Fourier test, two-level p-value: {ks_p_fourier:.4f}")

    ks_p_auto, _ = two_level_test(generate_bits, autocorrelation_test,
                                  n_samples=50, sample_size=10000)
    print(f"Good RNG - Autocorrelation (lag=1), two-level p-value: {ks_p_auto:.4f}")

    ks_p_gaps, _ = two_level_test(generate_bits, gaps_test,
                                  n_samples=50, sample_size=10000)
    print(f"Good RNG - Gaps test, two-level p-value: {ks_p_gaps:.4f}")

    print("3) Bad generator demo")
    bad_fourier, _ = two_level_test(bad_generate_bits, fourier_test,
                                    n_samples=50, sample_size=1024)
    print(f"Bad RNG  - Fourier test, two-level p-value: {bad_fourier:.4f}")

    bad_auto, _ = two_level_test(bad_generate_bits, autocorrelation_test,
                                 n_samples=50, sample_size=10000)
    print(f"Bad RNG  - Autocorrelation (lag=1), two-level p-value: {bad_auto:.4f}")

    bad_gaps, _ = two_level_test(bad_generate_bits, gaps_test,
                                 n_samples=50, sample_size=10000)
    print(f"Bad RNG  - Gaps test, two-level p-value: {bad_gaps:.4f}")

if __name__ == "__main__":
    main()

    