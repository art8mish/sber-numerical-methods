#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#if !defined(__x86_64__) && !defined(__i386__)
#error "Perftest requires x86 or x86_64 architecture for RDTSC"
#endif
#include <x86intrin.h>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#endif

extern "C" float logf(float x);
__m256 logf8_avx2(__m256 x);

namespace perftest {

using LogfFn = float (*)(float);

volatile float g_sink = 0.0f;

inline std::uint64_t read_tsc_start() {
    _mm_lfence();
    return __rdtsc();
}

inline std::uint64_t read_tsc_end() {
    unsigned aux = 0;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

inline void pin_thread_to_cpu0() {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

template <std::size_t LANES> std::uint64_t bench_scalar(std::size_t iters, LogfFn f) {
    static_assert(LANES >= 1);
    std::array<float, LANES> x{};
    for (std::size_t j = 0; j < LANES; ++j)
        x[j] = 1.001f + 0.1f * static_cast<float>(j);

    const std::uint64_t t1 = read_tsc_start();
    for (std::size_t i = 0; i < iters; ++i)
        for (std::size_t j = 0; j < LANES; ++j)
            x[j] = f(x[j] + 1.0f);
    const std::uint64_t t2 = read_tsc_end();

    float sum = 0.0f;
    for (float v : x)
        sum += v;
    g_sink = sum;
    return t2 - t1;
}

inline __m256 init_vec(float base) {
    return _mm256_setr_ps(base + 0.00f, base + 0.10f, base + 0.20f, base + 0.30f, base + 0.40f,
                          base + 0.50f, base + 0.60f, base + 0.70f);
}

inline float reduce_first(__m256 v) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    return buf[0];
}

template <std::size_t LANES> std::uint64_t bench_vector(std::size_t iters) {
    static_assert(LANES >= 1);
    alignas(32) __m256 x[LANES];
    for (std::size_t k = 0; k < LANES; ++k)
        x[k] = init_vec(1.001f + 0.05f * static_cast<float>(k));
    const __m256 one = _mm256_set1_ps(1.0f);

    const std::uint64_t t1 = read_tsc_start();
    for (std::size_t i = 0; i < iters; ++i)
        for (std::size_t k = 0; k < LANES; ++k)
            x[k] = logf8_avx2(_mm256_add_ps(x[k], one));
    const std::uint64_t t2 = read_tsc_end();

    __m256 acc = _mm256_setzero_ps();
    for (std::size_t k = 0; k < LANES; ++k)
        acc = _mm256_add_ps(acc, x[k]);
    g_sink = reduce_first(acc);
    return t2 - t1;
}

constexpr int WARMUP_RUNS = 4;
constexpr int MEASURE_RUNS = 13;

template <typename Kernel> std::vector<double> collect(std::size_t ops_per_run, Kernel kernel) {
    for (int i = 0; i < WARMUP_RUNS; ++i)
        (void)kernel();
    std::vector<double> samples;
    samples.reserve(MEASURE_RUNS);
    for (int i = 0; i < MEASURE_RUNS; ++i)
        samples.push_back(static_cast<double>(kernel()) / static_cast<double>(ops_per_run));
    return samples;
}

double median(std::string_view label, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const double med = samples[samples.size() / 2];
    const double avg = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double sq = 0.0;
    for (double x : samples)
        sq += (x - avg) * (x - avg);
    const double sd = std::sqrt(sq / samples.size());
    const double cv = avg != 0.0 ? 100.0 * sd / avg : 0.0;

    std::cout << label << ":\n\tsamples_cpe = [";
    for (std::size_t i = 0; i < samples.size(); ++i)
        std::cout << (i ? ", " : "") << std::format("{:.6f}", samples[i]);
    std::cout << std::format("]\n\tmedian_cpe  = {:.6f}\n\tmean_cpe    = {:.6f}\n"
                             "\tstddev_cpe  = {:.6f}\n\tcv_percent  = {:.6f}%\n\n",
                             med, avg, sd, cv);
    return med;
}

#if defined(__linux__)
LogfFn resolve_libm_logf() {
    dlerror();
    void *const p = dlsym(RTLD_NEXT, "logf");
    if (dlerror() != nullptr || p == nullptr)
        return nullptr;
    return reinterpret_cast<LogfFn>(p);
}
#else
LogfFn resolve_libm_logf() {
    return nullptr;
}
#endif

} // namespace perftest

namespace {

constexpr std::size_t ITERS = 5'000'000;
constexpr std::size_t SCALAR_LANES = 8;
constexpr std::size_t VECTOR_LANES = 4;
constexpr std::size_t VEC_WIDTH = 8;

} // namespace

int main() {
    using namespace perftest;
    pin_thread_to_cpu0();

    const LogfFn libm_logf = resolve_libm_logf();
    if (libm_logf == nullptr) {
        std::cerr << "perftest: dlsym(RTLD_NEXT, \"logf\") failed\n";
        return 1;
    }

    std::cout << std::format("Performance test: libm logf vs custom logf vs vector logf8_avx2\n"
                             "iters={}, warmup_runs={}, measure_runs={}, scalar_lanes={}, "
                             "vector_lanes={}\n\n",
                             ITERS, WARMUP_RUNS, MEASURE_RUNS, SCALAR_LANES, VECTOR_LANES);

    const double ml = median("Libm logf latency (CPE)",
                             collect(ITERS, [&] { return bench_scalar<1>(ITERS, libm_logf); }));
    const double mt = median("Libm logf throughput (CPE)", collect(ITERS * SCALAR_LANES, [&] {
                                 return bench_scalar<SCALAR_LANES>(ITERS, libm_logf);
                             }));
    const double ul = median("Custom logf latency (CPE)",
                             collect(ITERS, [&] { return bench_scalar<1>(ITERS, ::logf); }));
    const double ut = median("Custom logf throughput (CPE)", collect(ITERS * SCALAR_LANES, [&] {
                                 return bench_scalar<SCALAR_LANES>(ITERS, ::logf);
                             }));
    const double vl = median("Vector logf8_avx2 latency (CPE)",
                             collect(ITERS * VEC_WIDTH, [&] { return bench_vector<1>(ITERS); }));
    const double vt = median("Vector logf8_avx2 throughput (CPE)",
                             collect(ITERS * VECTOR_LANES * VEC_WIDTH,
                                     [&] { return bench_vector<VECTOR_LANES>(ITERS); }));

    std::cout << std::format(
        "Speedup vs vector (median CPE, lower is better; ratio = slower/faster):\n"
        "\tlatency:    libm={:.6f}  custom={:.6f}  vector={:.6f}\n"
        "\t            libm/vector={:.6f}  custom/vector={:.6f}\n"
        "\tthroughput: libm={:.6f}  custom={:.6f}  vector={:.6f}\n"
        "\t            libm/vector={:.6f}  custom/vector={:.6f}\n",
        ml, ul, vl, ml / vl, ul / vl, mt, ut, vt, mt / vt, ut / vt);

    return 0;
}
