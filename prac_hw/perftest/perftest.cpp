#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if !defined(__x86_64__) && !defined(__i386__)
#error "Perftest requires x86 or x86_64 architecture for RDTSC"
#endif

#include <x86intrin.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace perftest {

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

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1u) == 1u) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline double mean(const std::vector<double> &values) {
    const double s = std::accumulate(values.begin(), values.end(), 0.0);
    return s / static_cast<double>(values.size());
}

inline double stddev(const std::vector<double> &values, double avg) {
    double sq = 0.0;
    for (double x : values) {
        const double d = x - avg;
        sq += d * d;
    }
    return std::sqrt(sq / static_cast<double>(values.size()));
}

template <typename F> std::uint64_t cycles_latency(std::size_t iters, F &&f) {
    auto &&fn = std::forward<F>(f);
    float x = 1.2345f;
    const std::uint64_t t1 = read_tsc_start();
    for (std::size_t i = 0; i < iters; ++i) {
        x = fn(x + 1.0f);
    }
    const std::uint64_t t2 = read_tsc_end();
    g_sink = x;
    return t2 - t1;
}

template <typename F>
std::uint64_t cycles_throughput(std::size_t iters, std::size_t throughput_lanes, F &&f) {
    assert(throughput_lanes > 0);
    auto &&fn = std::forward<F>(f);
    std::vector<float> x(throughput_lanes);
    for (std::size_t j = 0; j < throughput_lanes; ++j) {
        x[j] = 1.001f + 0.1f * static_cast<float>(j);
    }
    const std::uint64_t t1 = read_tsc_start();
    for (std::size_t i = 0; i < iters; ++i) {
        for (std::size_t j = 0; j < throughput_lanes; ++j) {
            x[j] = fn(x[j] + 1.0f);
        }
    }
    const std::uint64_t t2 = read_tsc_end();
    float sum = 0.0f;
    for (float v : x) {
        sum += v;
    }
    g_sink = sum;
    return t2 - t1;
}

template <typename F>
std::vector<double> measure_cpe(std::size_t iters, int warmup_runs, int measure_runs, bool latency,
                                std::size_t throughput_lanes, F &&f) {
    assert(latency || throughput_lanes > 0);
    const std::size_t ops_per_run = latency ? iters : iters * throughput_lanes;

    std::vector<double> cpe_samples;
    cpe_samples.reserve(static_cast<std::size_t>(measure_runs));

    for (int i = 0; i < warmup_runs; ++i) {
        if (latency)
            cycles_latency(iters, f);
        else
            cycles_throughput(iters, throughput_lanes, f);
    }

    for (int i = 0; i < measure_runs; ++i) {
        const std::uint64_t cycles =
            latency ? cycles_latency(iters, f) : cycles_throughput(iters, throughput_lanes, f);
        cpe_samples.push_back(static_cast<double>(cycles) / static_cast<double>(ops_per_run));
    }

    return cpe_samples;
}

inline void print_stats(const std::string &label, const std::vector<double> &cpe_samples) {
    const double med = median(cpe_samples);
    const double avg = mean(cpe_samples);
    const double sd = stddev(cpe_samples, avg);
    const double cv_percent = (avg != 0.0) ? (100.0 * sd / avg) : 0.0;

    std::cout << label << ":\n";
    std::cout << "\tsamples_cpe = [";
    for (std::size_t i = 0; i < cpe_samples.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << cpe_samples[i];
    }
    std::cout << "]\n";
    std::cout << "\tmedian_cpe  = " << med << "\n";
    std::cout << "\tmean_cpe    = " << avg << "\n";
    std::cout << "\tstddev_cpe  = " << sd << "\n";
    std::cout << "\tcv_percent  = " << cv_percent << "%\n";
}

} // namespace perftest

namespace {

constexpr std::size_t ITERS = 20'000'000;
constexpr int WARMUP_RUNS = 4;
constexpr int MEASURE_RUNS = 13;
constexpr std::size_t THROUGHPUT_LANES = 8;

} // namespace

int main() {
    using namespace perftest;

    std::cout << std::fixed << std::setprecision(6);
    pin_thread_to_cpu0();

    const auto f = [](float x) { return std::sqrt(x); };

    const std::vector<double> latency_samples =
        measure_cpe(ITERS, WARMUP_RUNS, MEASURE_RUNS, true, 0, f);
    const std::vector<double> throughput_samples =
        measure_cpe(ITERS, WARMUP_RUNS, MEASURE_RUNS, false, THROUGHPUT_LANES, f);

    std::cout << "Performance test\n";
    std::cout << "iters=" << ITERS << ", warmup_runs=" << WARMUP_RUNS << ", measure_runs=" << MEASURE_RUNS
              << ", throughput_lanes=" << THROUGHPUT_LANES << "\n\n";

    print_stats("Latency (CPE)", latency_samples);
    std::cout << "\n";
    print_stats("Throughput (CPE)", throughput_samples);

    return 0;
}
