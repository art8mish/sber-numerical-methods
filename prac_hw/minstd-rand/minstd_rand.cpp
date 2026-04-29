#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

constexpr uint32_t A = 48271u;
constexpr uint32_t M = 2147483647u; // 2^31 - 1
constexpr size_t BLOCK_SIZE = 16;
volatile uint32_t RUNTIME_ZERO = 0u;

uint32_t normalize_seed(uint32_t seed) {
    seed %= M;
    return seed == 0u ? 1u : seed;
}

uint32_t mul_mod(uint32_t x, uint32_t y) {
    uint64_t z = static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
    z = (z >> 31) + (z & M);
    z = (z >> 31) + (z & M);
    if (z >= M) {
        z -= M;
    }
    return static_cast<uint32_t>(z);
}

double to_uniform_pm1(uint32_t x) {
    return 2.0 * (static_cast<double>(x) / static_cast<double>(M)) - 1.0;
}

uint64_t monte_carlo_scalar(size_t n_points, uint32_t seed) {
    uint32_t state = normalize_seed(seed) ^ RUNTIME_ZERO;
    uint64_t hits = 0;

    for (size_t i = 0; i < n_points; ++i) {
        state = mul_mod(A, state);
        const double u1 = to_uniform_pm1(state);
        state = mul_mod(A, state);
        const double u2 = to_uniform_pm1(state);
        if (u1 * u1 + u2 * u2 <= 1.0) {
            ++hits;
        }
    }
    return hits;
}

uint64_t monte_carlo_blocked(size_t n_points, uint32_t seed) {
    uint32_t state = normalize_seed(seed) ^ RUNTIME_ZERO;
    std::array<uint32_t, BLOCK_SIZE> powers{};
    powers[0] = A;
    for (size_t i = 1; i < BLOCK_SIZE; ++i) {
        powers[i] = mul_mod(powers[i - 1], A);
    }

    uint64_t hits = 0;
    size_t produced = 0;
    std::array<uint32_t, BLOCK_SIZE> block{};

    while (produced < n_points) {
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            block[i] = mul_mod(powers[i], state);
        }
        state = block[BLOCK_SIZE - 1];

        for (size_t i = 0; i + 1 < BLOCK_SIZE && produced < n_points; i += 2) {
            const double u1 = to_uniform_pm1(block[i]);
            const double u2 = to_uniform_pm1(block[i + 1]);
            if (u1 * u1 + u2 * u2 <= 1.0) {
                ++hits;
            }
            ++produced;
        }
    }
    return hits;
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

struct RunResult {
    uint64_t hits;
    uint64_t cycles;
};

RunResult run_scalar_once(size_t n_points, uint32_t seed) {
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t t1 = read_tsc_start();
    const uint64_t hits = monte_carlo_scalar(n_points, seed);
    const uint64_t t2 = read_tsc_end();
    return {hits, t2 - t1};
#else
    return {monte_carlo_scalar(n_points, seed), 0};
#endif
}

RunResult run_blocked_once(size_t n_points, uint32_t seed) {
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t t1 = read_tsc_start();
    const uint64_t hits = monte_carlo_blocked(n_points, seed);
    const uint64_t t2 = read_tsc_end();
    return {hits, t2 - t1};
#else
    return {monte_carlo_blocked(n_points, seed), 0};
#endif
}

int main() {
    constexpr uint32_t seed = 777u;
    const size_t n_points = 20'000'000;

    RunResult best_scalar{0, UINT64_MAX};
    RunResult best_blocked{0, UINT64_MAX};
    for (int i = 0; i < 3; ++i) {
        const RunResult s = run_scalar_once(n_points, seed);
        if (s.cycles < best_scalar.cycles) {
            best_scalar = s;
        }
        const RunResult b = run_blocked_once(n_points, seed);
        if (b.cycles < best_blocked.cycles) {
            best_blocked = b;
        }
    }

    const uint64_t hits_scalar = best_scalar.hits;
    const uint64_t hits_blocked = best_blocked.hits;
    const uint64_t cycles_scalar = best_scalar.cycles;
    const uint64_t cycles_blocked = best_blocked.cycles;

    const double pi_scalar = 4.0 * static_cast<double>(hits_scalar) / static_cast<double>(n_points);
    const double pi_blocked = 4.0 * static_cast<double>(hits_blocked) / static_cast<double>(n_points);
    const double speedup = cycles_blocked > 0
                               ? static_cast<double>(cycles_scalar) / static_cast<double>(cycles_blocked)
                               : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "minstd_rand Monte-Carlo pi benchmark\n";
    std::cout << "Parameters: a=48271, m=2^31-1, seed=" << seed << ", N=" << n_points << "\n";
    std::cout << "Scalar : hits=" << hits_scalar << ", pi=" << pi_scalar << ", cycles=" << cycles_scalar
              << "\n";
    std::cout << "Blocked: hits=" << hits_blocked << ", pi=" << pi_blocked
              << ", cycles=" << cycles_blocked << ", block_size=" << BLOCK_SIZE << "\n";
    std::cout << "Speedup : " << speedup << "\n";
    std::cout << "Abs diff: " << std::abs(pi_scalar - pi_blocked) << "\n";
    return 0;
}
