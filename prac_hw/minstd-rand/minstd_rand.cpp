// Домашнее задание: minstd (minstd_rand), вектор fp32, бенчмарк clocks/element,
// параллельный MC для pi (OpenMP) на 100M 2D-точек. Без сторонних реализаций ГСЧ.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#if defined(__AVX2__)
#include <immintrin.h>
#define MINSTD_HAS_AVX2 1
#endif
#endif

#if defined(MINSTD_HAS_AVX2)
static inline __m256 u32x8_to_ps(__m256i u) {
    return _mm256_cvtepi32_ps(u);
}
#endif

// ---------------------------------------------------------------------------
// Ядро LCG: совпадает с std::minstd_rand (a=48271, c=0, m=2^31-1)
// ---------------------------------------------------------------------------

constexpr uint32_t A = 48271u;
constexpr uint32_t M = 2147483647u; // 2^31 - 1
constexpr size_t kBatchLanes = 8;   // ширина батча a^k (степень двойки)
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

// a^e mod m, m = 2^31-1; для прыжка в потоке
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

// Состояние после пропуска k целочисленных выходов генератора,
// начиная с нормализованного seed (как после seed(s) у std::minstd_rand).
uint32_t state_after_skip(uint32_t normalized_seed, uint64_t skip_outputs) {
    const uint32_t mult = pow_mod_a(skip_outputs);
    return mul_mod(mult, normalized_seed);
}

double to_uniform_pm1(uint32_t x) {
    return 2.0 * (static_cast<double>(x) / static_cast<double>(M)) - 1.0;
}

// Одно значение std::minstd_rand::operator(): s <- A*s mod M, возвращается новое s.
uint32_t next_u32(uint32_t& state) {
    state = mul_mod(A, state);
    return state;
}

constexpr float INV_M_F = static_cast<float>(1.0 / static_cast<double>(M));

float u32_to_uniform11_fp32(uint32_t x) {
    return (static_cast<float>(x) * INV_M_F) * 2.0f - 1.0f;
}

float u32_to_uniform01_fp32(uint32_t x) {
    return static_cast<float>(x) * INV_M_F;
}

// ---------------------------------------------------------------------------
// Задача 1: последовательная и векторная (AVX2) генерация fp32 на [0,1).
// n — степень двойки, как в задании.
// ---------------------------------------------------------------------------

void gen_uniform01_fp32_scalar(float* dst, size_t n, uint32_t seed) {
    uint32_t state = normalize_seed(seed) ^ RUNTIME_ZERO;
    for (size_t i = 0; i < n; ++i) {
        state = mul_mod(A, state);
        dst[i] = u32_to_uniform01_fp32(state);
    }
}

void gen_uniform01_fp32_scalar_cont(float* dst, size_t n, uint32_t& pre_state) {
    for (size_t i = 0; i < n; ++i) {
        pre_state = mul_mod(A, pre_state);
        dst[i] = u32_to_uniform01_fp32(pre_state);
    }
}

#if defined(MINSTD_HAS_AVX2)

void gen_uniform01_fp32_avx2(float* dst, size_t n, uint32_t seed) {
    if ((n % kBatchLanes) != 0 || n == 0) {
        gen_uniform01_fp32_scalar(dst, n, seed);
        return;
    }
    std::vector<uint32_t> lane_pow(kBatchLanes);
    lane_pow[0] = A;
    for (size_t i = 1; i < kBatchLanes; ++i) {
        lane_pow[i] = mul_mod(lane_pow[i - 1], A);
    }
    const uint32_t step_mult = lane_pow[kBatchLanes - 1]; // A^8

    uint32_t base_state = normalize_seed(seed) ^ RUNTIME_ZERO;
    size_t off = 0;
    const __m256 invm = _mm256_set1_ps(INV_M_F);

    while (off < n) {
        const uint32_t base = base_state;

        alignas(32) uint32_t xs[kBatchLanes];
        for (size_t j = 0; j < kBatchLanes; ++j) {
            xs[j] = mul_mod(lane_pow[j], base);
        }
        base_state = mul_mod(step_mult, base);

        const __m256i xi = _mm256_load_si256(reinterpret_cast<const __m256i*>(xs));
        const __m256 f = _mm256_mul_ps(u32x8_to_ps(xi), invm);
        _mm256_storeu_ps(dst + off, f);
        off += kBatchLanes;
    }
}

void gen_uniform01_fp32_avx2_cont(float* dst, size_t n, uint32_t& pre_state,
                                  const std::vector<uint32_t>& lane_pow, uint32_t step_mult) {
    size_t off = 0;
    const __m256 invm = _mm256_set1_ps(INV_M_F);
    while (off < n) {
        const uint32_t base = pre_state;

        alignas(32) uint32_t xs[kBatchLanes];
        for (size_t j = 0; j < kBatchLanes; ++j) {
            xs[j] = mul_mod(lane_pow[j], base);
        }
        pre_state = mul_mod(step_mult, base);

        const __m256i xi = _mm256_load_si256(reinterpret_cast<const __m256i*>(xs));
        const __m256 f = _mm256_mul_ps(u32x8_to_ps(xi), invm);
        _mm256_storeu_ps(dst + off, f);
        off += kBatchLanes;
    }
}

#endif

static thread_local std::vector<uint32_t> g_lane_pow;
static thread_local uint32_t g_step_mult = 0;

static void ensure_lane_powers() {
    if (g_step_mult != 0) {
        return;
    }
    g_lane_pow.resize(kBatchLanes);
    g_lane_pow[0] = A;
    for (size_t i = 1; i < kBatchLanes; ++i) {
        g_lane_pow[i] = mul_mod(g_lane_pow[i - 1], A);
    }
    g_step_mult = g_lane_pow[kBatchLanes - 1];
}

// Потоковая генерация U[0,1): pre_state — внутреннее состояние до следующего выхода (как у std после seed).
void gen_uniform01_fp32_vector_cont(float* dst, size_t n, uint32_t& pre_state) {
#if defined(MINSTD_HAS_AVX2)
    ensure_lane_powers();
    const size_t bulk = n - (n % kBatchLanes);
    if (bulk > 0) {
        gen_uniform01_fp32_avx2_cont(dst, bulk, pre_state, g_lane_pow, g_step_mult);
    }
    gen_uniform01_fp32_scalar_cont(dst + bulk, n - bulk, pre_state);
#else
    gen_uniform01_fp32_scalar_cont(dst, n, pre_state);
#endif
}

void gen_uniform01_fp32_vector(float* dst, size_t n, uint32_t seed) {
#if defined(MINSTD_HAS_AVX2)
    gen_uniform01_fp32_avx2(dst, n, seed);
#else
    gen_uniform01_fp32_scalar(dst, n, seed);
#endif
}

bool verify_bitstream_matches_std(size_t count) {
    uint32_t s = normalize_seed(777u);
    std::minstd_rand ref(777u);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t got = next_u32(s);
        const uint32_t want = static_cast<uint32_t>(ref());
        if (got != want) {
            std::cerr << "Mismatch at " << i << ": got " << got << " want " << want << "\n";
            return false;
        }
    }
    return true;
}

bool verify_fp32_sequence(size_t n, uint32_t seed) {
    std::vector<float> a(n), b(n);
    gen_uniform01_fp32_scalar(a.data(), n, seed);
    gen_uniform01_fp32_vector(b.data(), n, seed);
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            std::cerr << "fp32 scalar vs vector mismatch at " << i << ": " << a[i] << " vs " << b[i]
                      << "\n";
            return false;
        }
    }
    return true;
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

// ---------------------------------------------------------------------------
// Задача 2: параллельный Monte Carlo для pi, 100M 2D-векторов (точек),
// используется векторный fp32-путь для локальных чанков не обязателен —
// используем правильный jump-ahead и fp32 u(-1,1).
// ---------------------------------------------------------------------------

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
        size_t i = 0;
        for (; i + 1 < block_size && produced < n_points; i += 2) {
            const uint32_t x1 = mul_mod(powers[i], base);
            const uint32_t x2 = mul_mod(powers[i + 1], base);
            const double u1 = to_uniform_pm1(x1);
            const double u2 = to_uniform_pm1(x2);
            if (u1 * u1 + u2 * u2 <= 1.0) {
                ++hits;
            }
            ++produced;
        }
        const size_t pairs = i / 2;
        if (pairs > 0) {
            state = mul_mod(powers[2 * pairs - 1], base);
        }
    }
    return hits;
}

uint64_t monte_carlo_pi_fp32_omp(size_t n_points, uint32_t seed) {
    uint64_t hits = 0;
    const uint32_t s0 = normalize_seed(seed);

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : hits)
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t chunk = (n_points + static_cast<size_t>(nt) - 1u) / static_cast<size_t>(nt);
        const size_t start = static_cast<size_t>(tid) * chunk;
        const size_t end = std::min(start + chunk, n_points);

        uint32_t state = state_after_skip(s0, 2u * start);
        ensure_lane_powers();

        alignas(32) uint32_t xs[kBatchLanes];
        size_t i = start;
        while (i < end) {
            const size_t remain = end - i;
            if (remain >= 4) {
                const uint32_t base = state;
                for (size_t j = 0; j < kBatchLanes; ++j) {
                    xs[j] = mul_mod(g_lane_pow[j], base);
                }
                state = mul_mod(g_step_mult, base);
                for (size_t k = 0; k < 4; ++k) {
                    const float u1 = u32_to_uniform11_fp32(xs[2 * k]);
                    const float u2 = u32_to_uniform11_fp32(xs[2 * k + 1]);
                    if (u1 * u1 + u2 * u2 <= 1.0f) {
                        ++hits;
                    }
                }
                i += 4;
            } else {
                for (; i < end; ++i) {
                    state = mul_mod(A, state);
                    const float u1 = u32_to_uniform11_fp32(state);
                    state = mul_mod(A, state);
                    const float u2 = u32_to_uniform11_fp32(state);
                    if (u1 * u1 + u2 * u2 <= 1.0f) {
                        ++hits;
                    }
                }
            }
        }
    }
#else
    uint32_t state = s0;
    ensure_lane_powers();
    alignas(32) uint32_t xs[kBatchLanes];
    size_t i = 0;
    while (i < n_points) {
        const size_t remain = n_points - i;
        if (remain >= 4) {
            const uint32_t base = state;
            for (size_t j = 0; j < kBatchLanes; ++j) {
                xs[j] = mul_mod(g_lane_pow[j], base);
            }
            state = mul_mod(g_step_mult, base);
            for (size_t k = 0; k < 4; ++k) {
                const float u1 = u32_to_uniform11_fp32(xs[2 * k]);
                const float u2 = u32_to_uniform11_fp32(xs[2 * k + 1]);
                if (u1 * u1 + u2 * u2 <= 1.0f) {
                    ++hits;
                }
            }
            i += 4;
        } else {
            for (; i < n_points; ++i) {
                state = mul_mod(A, state);
                const float u1 = u32_to_uniform11_fp32(state);
                state = mul_mod(A, state);
                const float u2 = u32_to_uniform11_fp32(state);
                if (u1 * u1 + u2 * u2 <= 1.0f) {
                    ++hits;
                }
            }
        }
    }
#endif
    return hits;
}

std::vector<uint32_t> make_powers(size_t block_size) {
    std::vector<uint32_t> powers(block_size);
    powers[0] = A;
    for (size_t i = 1; i < block_size; ++i) {
        powers[i] = mul_mod(powers[i - 1], A);
    }
    return powers;
}

struct RunResult {
    uint64_t hits;
    uint64_t cycles;
};

RunResult run_scalar_once(size_t n_points, uint32_t seed) {
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t t1 = read_tsc_start();
    const uint64_t h = monte_carlo_scalar(n_points, seed);
    const uint64_t t2 = read_tsc_end();
    return {h, t2 - t1};
#else
    return {monte_carlo_scalar(n_points, seed), 0};
#endif
}

RunResult run_blocked_once(size_t n_points, uint32_t seed, const std::vector<uint32_t>& powers) {
#if defined(__x86_64__) || defined(__i386__)
    const uint64_t t1 = read_tsc_start();
    const uint64_t h = monte_carlo_blocked(n_points, seed, powers);
    const uint64_t t2 = read_tsc_end();
    return {h, t2 - t1};
#else
    return {monte_carlo_blocked(n_points, seed, powers), 0};
#endif
}

struct BenchStats {
    uint64_t hits;
    uint64_t min_cycles;
    uint64_t median_cycles;
};

template <typename Runner>
BenchStats benchmark_seq(Runner run_once, int repeat) {
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

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--bench-double] [--bench-fp32] [--verify] [--pi-omp]\n"
              << "  (default runs verify + fp32 bench + double blocked bench + pi OpenMP)\n";
}

int main(int argc, char** argv) {
    bool do_verify = true;
    bool do_bench_fp32 = true;
    bool do_bench_double = true;
    bool do_pi_omp = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--verify-only") {
            do_bench_fp32 = false;
            do_bench_double = false;
            do_pi_omp = false;
        } else if (arg == "--bench-double") {
            do_verify = false;
            do_bench_fp32 = false;
            do_pi_omp = false;
            do_bench_double = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    constexpr uint32_t seed = 777u;
    constexpr int repeat = 7;

    std::cout << std::fixed << std::setprecision(6);

    // --- Проверка соответствия std::minstd_rand и fp32 scalar/vector ---
    if (do_verify) {
        constexpr size_t verify_n = 1u << 20;
        const bool ok_bits = verify_bitstream_matches_std(verify_n);
        const bool ok_fp = verify_fp32_sequence(65536u, seed);
        std::cout << "=== Verification (Task 1) ===\n";
        std::cout << "uint32 stream vs std::minstd_rand (" << verify_n << " steps): "
                  << (ok_bits ? "PASS" : "FAIL") << "\n";
        std::cout << "fp32 scalar vs vector (65536 floats): " << (ok_fp ? "PASS" : "FAIL") << "\n";
#if defined(MINSTD_HAS_AVX2)
        std::cout << "AVX2 fp32 path: enabled\n";
#else
        std::cout << "AVX2 fp32 path: disabled (scalar fallback)\n";
#endif
        if (!ok_bits || !ok_fp) {
            return 1;
        }
    }

    // --- clocks / element для fp32 ---
    if (do_bench_fp32) {
        constexpr size_t gen_n = 1u << 24; // степень двойки
        std::vector<float> buf(gen_n);

#if defined(__x86_64__) || defined(__i386__)
        std::vector<uint64_t> c_scal(repeat), c_vec(repeat);
        for (int r = 0; r < repeat; ++r) {
            const uint64_t t1 = read_tsc_start();
            gen_uniform01_fp32_scalar(buf.data(), gen_n, seed);
            c_scal[r] = read_tsc_end() - t1;
        }
        for (int r = 0; r < repeat; ++r) {
            const uint64_t t1 = read_tsc_start();
            gen_uniform01_fp32_vector(buf.data(), gen_n, seed);
            c_vec[r] = read_tsc_end() - t1;
        }
        std::sort(c_scal.begin(), c_scal.end());
        std::sort(c_vec.begin(), c_vec.end());
        const uint64_t med_scal = c_scal[repeat / 2];
        const uint64_t med_vec = c_vec[repeat / 2];
        const double cpe_scal = static_cast<double>(med_scal) / static_cast<double>(gen_n);
        const double cpe_vec = static_cast<double>(med_vec) / static_cast<double>(gen_n);

        std::cout << "\n=== Task 1: fp32 uniform [0,1), N=" << gen_n << " (median of " << repeat
                  << " runs) ===\n";
        std::cout << "Scalar clocks/element: " << cpe_scal << " (median_cycles=" << med_scal
                  << ")\n";
        std::cout << "Vector clocks/element: " << cpe_vec << " (median_cycles=" << med_vec
                  << ")\n";
#endif
    }

    // --- старый бенчмарк double scalar vs blocked (лекции) ---
    if (do_bench_double) {
        const size_t n_points = 20'000'000;
        const std::vector<size_t> block_sizes = {8, 16, 32, 64, 512};

        const BenchStats scalar =
            benchmark_seq([&]() { return run_scalar_once(n_points, seed); }, repeat);
        CandidateResult best_blocked{0, {0, UINT64_MAX, UINT64_MAX}};

        for (size_t block_size : block_sizes) {
            if (block_size < 2 || (block_size % 2 != 0)) {
                continue;
            }
            const std::vector<uint32_t> powers = make_powers(block_size);
            const BenchStats cur =
                benchmark_seq([&]() { return run_blocked_once(n_points, seed, powers); }, repeat);
            if (cur.median_cycles < best_blocked.stats.median_cycles) {
                best_blocked = {block_size, cur};
            }
        }

        const double pi_scalar =
            4.0 * static_cast<double>(scalar.hits) / static_cast<double>(n_points);
        const double pi_blocked =
            4.0 * static_cast<double>(best_blocked.stats.hits) / static_cast<double>(n_points);
        const double speedup = best_blocked.stats.median_cycles > 0
                                   ? static_cast<double>(scalar.median_cycles) /
                                         static_cast<double>(best_blocked.stats.median_cycles)
                                   : 0.0;

        std::cout << "\n=== Double-precision blocked MC pi (lecture benchmark), N=" << n_points
                  << " ===\n";
        std::cout << "Scalar : pi=" << pi_scalar << ", median_cycles=" << scalar.median_cycles
                  << "\n";
        std::cout << "Blocked: pi=" << pi_blocked << ", median_cycles=" << best_blocked.stats.median_cycles
                  << ", best_block_size=" << best_blocked.block_size << "\n";
        std::cout << "Speedup: " << speedup << ", hits_equal="
                  << (scalar.hits == best_blocked.stats.hits ? "true" : "false") << "\n";
    }

    // --- Task 2: 100M точек, OpenMP, fp32 + тот же LCG через jump-ahead ---
    if (do_pi_omp) {
        constexpr size_t n_pi = 100'000'000;
#if defined(_OPENMP)
        std::cout << "\n=== Task 2: parallel pi (fp32), N=" << n_pi << " points, OpenMP threads="
                  << omp_get_max_threads() << " ===\n";
#else
        std::cout << "\n=== Task 2: parallel pi (fp32), N=" << n_pi
                  << " points (OpenMP not compiled — sequential) ===\n";
#endif
#if defined(__x86_64__) || defined(__i386__)
        const uint64_t t1 = read_tsc_start();
#endif
        const uint64_t h = monte_carlo_pi_fp32_omp(n_pi, seed);
#if defined(__x86_64__) || defined(__i386__)
        const uint64_t t2 = read_tsc_end();
        const double cpe = static_cast<double>(t2 - t1) / static_cast<double>(n_pi);
        std::cout << "cycles total=" << (t2 - t1) << ", clocks/point=" << cpe << "\n";
#endif
        const double pi_est = 4.0 * static_cast<double>(h) / static_cast<double>(n_pi);
        std::cout << "hits=" << h << ", pi≈" << pi_est << "\n";
    }

    return 0;
}
