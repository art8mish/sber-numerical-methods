#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#endif
#endif

constexpr uint32_t A = 48271u;
constexpr uint32_t M = 2147483647u; // 2^31 - 1
constexpr uint32_t DEFAULT_SEED = 777u;
constexpr size_t PI_VECTORS = 100'000'000;
constexpr size_t LANES = 8;
constexpr float INV_M_F = static_cast<float>(1.0 / static_cast<double>(M));

uint32_t normalize_seed(uint32_t seed) {
    seed %= M;
    return seed == 0u ? 1u : seed;
}

// (x * y) mod M
uint32_t mul_mod(uint32_t x, uint32_t y) {
    uint64_t z = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
    z = (z >> 31) + (z & M);
    z = (z >> 31) + (z & M);
    if (z >= M)
        z -= M;
    return static_cast<uint32_t>(z);
}

// A^e mod M
uint32_t pow_mod_a(uint64_t e) {
    uint32_t r = 1u;
    uint32_t b = A;
    while (e > 0) {
        if (e & 1ULL) {
            r = mul_mod(r, b);
        }
        b = mul_mod(b, b);
        e >>= 1ULL;
    }
    return r;
}

double to_uniform_pm1(uint32_t x) {
    return 2.0 * (static_cast<double>(x) / static_cast<double>(M)) - 1.0;
}

float to_uniform01_fp32(uint32_t x) {
    return static_cast<float>(x) * INV_M_F;
}

class MinStdRand {
    uint32_t state_;

public:
    explicit MinStdRand(uint32_t seed) : state_(normalize_seed(seed)) {}

    uint32_t next() {
        state_ = mul_mod(A, state_);
        return state_;
    }

    void skip(uint64_t outputs) {
        const uint32_t mult = pow_mod_a(outputs);
        state_ = mul_mod(mult, state_);
    }
};

std::vector<uint32_t> make_lane_powers() {
    std::vector<uint32_t> p(LANES);
    p[0] = A;
    for (size_t i = 1; i < LANES; ++i) {
        p[i] = mul_mod(p[i - 1], A);
    }
    return p;
}

void gen_uniform01_scalar(float *out, size_t n, uint32_t seed) {
    MinStdRand gen(seed);
    for (size_t i = 0; i < n; ++i) {
        out[i] = to_uniform01_fp32(gen.next());
    }
}

void gen_uniform01_vector(float *out, size_t n, uint32_t seed) {
    uint32_t state = normalize_seed(seed);
    const std::vector<uint32_t> p = make_lane_powers();
    const uint32_t step = p[LANES - 1]; // A^8

    size_t i = 0;
#if defined(HAS_AVX2)
    const __m256 invm = _mm256_set1_ps(INV_M_F);
    alignas(32) uint32_t xs[LANES];
    for (; i + LANES <= n; i += LANES) {
        const uint32_t base = state;
        for (size_t j = 0; j < LANES; ++j) {
            xs[j] = mul_mod(p[j], base);
        }
        state = mul_mod(step, base);

        const __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i *>(xs));
        const __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), invm);
        _mm256_storeu_ps(out + i, vf);
    }
#endif
    for (; i < n; ++i) {
        state = mul_mod(A, state);
        out[i] = to_uniform01_fp32(state);
    }
}

bool verify_matches_std(uint32_t seed, size_t n) {
    MinStdRand mine(seed);
    std::minstd_rand ref(seed);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t got = mine.next();
        const uint32_t want = static_cast<uint32_t>(ref());
        if (got != want) {
            std::cerr << "Mismatch at " << i << ": got=" << got << " want=" << want << "\n";
            return false;
        }
    }
    return true;
}

bool verify_vector_matches_scalar(uint32_t seed, size_t n) {
    std::vector<float> a(n), b(n);
    gen_uniform01_scalar(a.data(), n, seed);
    gen_uniform01_vector(b.data(), n, seed);
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            std::cerr << "Vector/scalar mismatch at " << i << ": " << b[i] << " vs " << a[i]
                      << "\n";
            return false;
        }
    }
    return true;
}

uint64_t estimate_pi_hits(MinStdRand &gen, size_t n_vectors) {
    uint64_t hits = 0;
    for (size_t i = 0; i < n_vectors; ++i) {
        const double u1 = to_uniform_pm1(gen.next());
        const double u2 = to_uniform_pm1(gen.next());
        const double d = u1 * u1 + u2 * u2;
        if (d <= 1.0) {
            ++hits;
        }
    }
    return hits;
}

uint64_t estimate_pi_hits_parallel(size_t n_vectors, uint32_t seed) {
    uint64_t total_hits = 0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_hits)
    {
        const size_t nt = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t chunk = (n_vectors + nt - 1u) / nt;
        const size_t start = tid * chunk;
        const size_t end = std::min(start + chunk, n_vectors);

        if (start < end) {
            MinStdRand gen(seed);
            gen.skip(2u * start);
            total_hits += estimate_pi_hits(gen, end - start);
        }
    }
#else
    MinStdRand gen(seed);
    total_hits = estimate_pi_hits(gen, n_vectors);
#endif
    return total_hits;
}

#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t read_tsc_start() {
    _mm_lfence();
    return __rdtsc();
}

static inline uint64_t read_tsc_end() {
    unsigned aux = 0;
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}
#endif

struct RunStats {
    uint64_t hits;
    uint64_t cycles;
    double pi;
};

#if defined(__x86_64__) || defined(__i386__)
template <typename Func> uint64_t benchmark_cycles(Func &&f) {
    const uint64_t t1 = read_tsc_start();
    f();
    const uint64_t t2 = read_tsc_end();
    return t2 - t1;
}
#endif

void print_generator_bench(size_t n, uint32_t seed) {
    std::vector<float> buf(n);
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t scalar_cycles =
        benchmark_cycles([&]() { gen_uniform01_scalar(buf.data(), n, seed); });
    const uint64_t vector_cycles =
        benchmark_cycles([&]() { gen_uniform01_vector(buf.data(), n, seed); });
    const double scalar_cpe = static_cast<double>(scalar_cycles) / static_cast<double>(n);
    const double vector_cpe = static_cast<double>(vector_cycles) / static_cast<double>(n);
    const double speedup_sv =
        vector_cycles > 0 ? static_cast<double>(scalar_cycles) / static_cast<double>(vector_cycles)
                          : 0.0;

    std::cout << "\nGenerator clocks/element (N=2^24)\n";
    std::cout << "scalar_cpe=" << scalar_cpe << " (cycles=" << scalar_cycles << ")\n";
    std::cout << "vector_cpe=" << vector_cpe << " (cycles=" << vector_cycles << ")\n";
    std::cout << "Speedup (scalar/vector generator): " << speedup_sv << "\n";
#else
    (void)seed;
    std::cout << "\nGenerator clocks/element: unavailable on this architecture\n";
#endif
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const bool same_as_std = verify_matches_std(DEFAULT_SEED, 1u << 20);
    const bool vec_ok = verify_vector_matches_scalar(DEFAULT_SEED, 1u << 16);
    if (!same_as_std || !vec_ok) {
        std::cout << "Verification with std::minstd_rand: FAIL\n";
        return 1;
    }

    std::cout << "Verification with std::minstd_rand: PASS\n";
    std::cout << "Benchmark vectors: " << PI_VECTORS << "\n";
#if defined(_OPENMP)
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP threads: unavailable (compiled without -fopenmp)\n";
#endif

    print_generator_bench(1u << 24, DEFAULT_SEED);

    uint64_t scalar_hits = 0;
    uint64_t parallel_hits = 0;
    uint64_t scalar_cycles = 0;
    uint64_t parallel_cycles = 0;

    MinStdRand scalar_gen(DEFAULT_SEED);
#if defined(__x86_64__) || defined(__i386__)
    scalar_cycles =
        benchmark_cycles([&]() { scalar_hits = estimate_pi_hits(scalar_gen, PI_VECTORS); });
    parallel_cycles = benchmark_cycles(
        [&]() { parallel_hits = estimate_pi_hits_parallel(PI_VECTORS, DEFAULT_SEED); });
#else
    scalar_hits = estimate_pi_hits(scalar_gen, PI_VECTORS);
    parallel_hits = estimate_pi_hits_parallel(PI_VECTORS, DEFAULT_SEED);
#endif
    const double scalar_pi =
        4.0 * static_cast<double>(scalar_hits) / static_cast<double>(PI_VECTORS);
    const double parallel_pi =
        4.0 * static_cast<double>(parallel_hits) / static_cast<double>(PI_VECTORS);

    const double speedup = parallel_cycles > 0 ? static_cast<double>(scalar_cycles) /
                                                     static_cast<double>(parallel_cycles)
                                               : 0.0;

    std::cout << "\nMonte Carlo benchmark results:\n";
    std::cout << "Case      | hits      | pi       | cycles\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "scalar    | " << scalar_hits << " | " << scalar_pi << " | " << scalar_cycles
              << "\n";
    std::cout << "parallel  | " << parallel_hits << " | " << parallel_pi << " | " << parallel_cycles
              << "\n";

    std::cout << "\nSpeedup (scalar/parallel): " << speedup << "\n";
    std::cout << "Abs(pi_scalar - pi_parallel): " << std::abs(scalar_pi - parallel_pi) << "\n";
    std::cout << "Hits equal: " << (scalar_hits == parallel_hits ? "true" : "false") << "\n";
#if defined(HAS_AVX2)
    std::cout << "Vectorization note: AVX2 is enabled for fp32 conversion/store path.\n";
#else
    std::cout
        << "Vectorization note: AVX2 unavailable, vector path falls back to scalar operations.\n";
#endif
    return 0;
}
