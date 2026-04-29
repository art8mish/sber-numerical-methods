#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

constexpr uint32_t A = 48271u;
constexpr uint32_t M = 2147483647u; // 2^31 - 1
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

uint64_t monte_carlo_blocked(size_t n_points, uint32_t seed, const std::vector<uint32_t>& powers) {
    uint32_t state = normalize_seed(seed) ^ RUNTIME_ZERO;
    const size_t block_size = powers.size();
    uint64_t hits = 0;
    size_t produced = 0;

    while (produced < n_points) {
        const uint32_t base = state;
        state = mul_mod(powers[block_size - 1], base);

        for (size_t i = 0; i + 1 < block_size && produced < n_points; i += 2) {
            const uint32_t x1 = mul_mod(powers[i], base);
            const uint32_t x2 = mul_mod(powers[i + 1], base);
            const double u1 = to_uniform_pm1(x1);
            const double u2 = to_uniform_pm1(x2);
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

RunResult run_blocked_once(size_t n_points, uint32_t seed, const std::vector<uint32_t>& powers) {
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t t1 = read_tsc_start();
    const uint64_t hits = monte_carlo_blocked(n_points, seed, powers);
    const uint64_t t2 = read_tsc_end();
    return {hits, t2 - t1};
#else
    return {monte_carlo_blocked(n_points, seed, powers), 0};
#endif
}

std::vector<uint32_t> make_powers(size_t block_size) {
    std::vector<uint32_t> powers(block_size);
    powers[0] = A;
    for (size_t i = 1; i < block_size; ++i) {
        powers[i] = mul_mod(powers[i - 1], A);
    }
    return powers;
}

struct BenchStats {
    uint64_t hits;
    uint64_t min_cycles;
    uint64_t median_cycles;
};

template <typename Runner>
BenchStats benchmark(Runner run_once, int repeat) {
    std::vector<uint64_t> cycles;
    cycles.reserve(static_cast<size_t>(repeat));
    uint64_t hits = 0;

    for (int i = 0; i < repeat; ++i) {
        const RunResult run = run_once();
        hits = run.hits;
        cycles.push_back(run.cycles);
    }

    std::sort(cycles.begin(), cycles.end());
    const uint64_t min_cycles = cycles.front();
    const uint64_t median_cycles = cycles[cycles.size() / 2];
    return {hits, min_cycles, median_cycles};
}

struct CandidateResult {
    size_t block_size;
    BenchStats stats;
};

int main() {
    constexpr uint32_t seed = 777u;
    const size_t n_points = 20'000'000;
    constexpr int repeat = 7;
    const std::vector<size_t> block_sizes = {8, 16, 32, 64};

    const BenchStats scalar = benchmark([&]() { return run_scalar_once(n_points, seed); }, repeat);
    CandidateResult best_blocked{0, {0, UINT64_MAX, UINT64_MAX}};

    for (size_t block_size : block_sizes) {
        if (block_size < 2 || (block_size % 2 != 0)) {
            continue;
        }
        const std::vector<uint32_t> powers = make_powers(block_size);
        const BenchStats cur =
            benchmark([&]() { return run_blocked_once(n_points, seed, powers); }, repeat);
        if (cur.median_cycles < best_blocked.stats.median_cycles) {
            best_blocked = {block_size, cur};
        }
    }

    if (best_blocked.block_size == 0) {
        std::cerr << "No valid blocked configuration\n";
        return 1;
    }

    const uint64_t hits_scalar = scalar.hits;
    const uint64_t hits_blocked = best_blocked.stats.hits;
    const uint64_t cycles_scalar = scalar.median_cycles;
    const uint64_t cycles_blocked = best_blocked.stats.median_cycles;

    const double pi_scalar = 4.0 * static_cast<double>(hits_scalar) / static_cast<double>(n_points);
    const double pi_blocked = 4.0 * static_cast<double>(hits_blocked) / static_cast<double>(n_points);
    const double speedup = cycles_blocked > 0
                               ? static_cast<double>(cycles_scalar) / static_cast<double>(cycles_blocked)
                               : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "minstd_rand Monte-Carlo pi benchmark\n";
    std::cout << "Parameters: a=48271, m=2^31-1, seed=" << seed << ", N=" << n_points
              << ", repeats=" << repeat << "\n";
    std::cout << "Scalar : hits=" << hits_scalar << ", pi=" << pi_scalar
              << ", median_cycles=" << scalar.median_cycles << ", min_cycles=" << scalar.min_cycles
              << "\n";
    std::cout << "Blocked: hits=" << hits_blocked << ", pi=" << pi_blocked
              << ", median_cycles=" << cycles_blocked << ", min_cycles=" << best_blocked.stats.min_cycles
              << ", best_block_size=" << best_blocked.block_size << "\n";
    std::cout << "Speedup: " << speedup << "\n";
    std::cout << "Hits equal: " << (hits_scalar == hits_blocked ? "true" : "false") << "\n";
    std::cout << "Abs diff: " << std::abs(pi_scalar - pi_blocked) << "\n";
    return 0;
}
