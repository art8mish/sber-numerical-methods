#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int NUM_OPTIONS = 100;
constexpr std::int64_t PATHS_PER_OPTION = 100'000;
constexpr int BLOCK_SIZE = 256;

inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double black_scholes_call(double s0, double k, double r, double sigma, double tau) {
    if (tau <= 0.0) {
        return std::max(s0 - k, 0.0);
    }
    const double sig_sqrt_t = sigma * std::sqrt(tau);
    const double y_plus =
        (std::log(s0 / k) + tau * (r + 0.5 * sigma * sigma)) / sig_sqrt_t;
    const double y_minus =
        (std::log(s0 / k) + tau * (r - 0.5 * sigma * sigma)) / sig_sqrt_t;
    return s0 * norm_cdf(y_plus) - k * std::exp(-r * tau) * norm_cdf(y_minus);
}

struct OptionParams {
    double s0{};
    double k{};
    double r{};
    double sigma{};
    double t{};
};

void generate_params(std::uint64_t master_seed, std::vector<OptionParams>& out) {
    out.resize(NUM_OPTIONS);
    std::mt19937_64 gen(master_seed);
    std::uniform_real_distribution<double> u_s0(80.0, 120.0);
    std::uniform_real_distribution<double> u_k(80.0, 120.0);
    std::uniform_real_distribution<double> u_r(0.01, 0.08);
    std::uniform_real_distribution<double> u_sig(0.05, 0.40);
    std::uniform_real_distribution<double> u_t(0.25, 2.0);
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        out[i].s0 = u_s0(gen);
        out[i].k = u_k(gen);
        out[i].r = u_r(gen);
        out[i].sigma = u_sig(gen);
        out[i].t = u_t(gen);
    }
}

double mc_call_exact_sum(
    const OptionParams& p,
    std::uint64_t path_seed,
    std::vector<double>& z_block
) {
    std::mt19937_64 gen(path_seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double mu = (p.r - 0.5 * p.sigma * p.sigma) * p.t;
    const double vol = p.sigma * std::sqrt(p.t);

    double sum_payoff = 0.0;

    for (std::int64_t base = 0; base < PATHS_PER_OPTION; base += BLOCK_SIZE) {
        const int n = static_cast<int>(std::min<std::int64_t>(BLOCK_SIZE, PATHS_PER_OPTION - base));
        for (int j = 0; j < n; ++j) {
            z_block[static_cast<std::size_t>(j)] = normal(gen);
        }

        double block_sum = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : block_sum)
#endif
        for (int j = 0; j < n; ++j) {
            const double z = z_block[static_cast<std::size_t>(j)];
            const double st = p.s0 * std::exp(mu + vol * z);
            const double diff = st - p.k;
            block_sum += (diff > 0.0) ? diff : 0.0;
        }
        sum_payoff += block_sum;
    }
    return sum_payoff;
}

}  // namespace

int main() {
    constexpr std::uint64_t MASTER_SEED = 42;

    std::vector<OptionParams> params;
    generate_params(MASTER_SEED, params);

    std::vector<double> analytical(NUM_OPTIONS);
    std::vector<double> mc_price(NUM_OPTIONS);
    std::vector<double> abs_err(NUM_OPTIONS);

    const auto t0 = std::chrono::high_resolution_clock::now();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        const OptionParams& p = params[static_cast<std::size_t>(i)];
        analytical[i] = black_scholes_call(p.s0, p.k, p.r, p.sigma, p.t);

        const std::uint64_t path_seed =
            MASTER_SEED * 1'000'003ULL + static_cast<std::uint64_t>(i) * 97'621ULL + 17ULL;

        std::vector<double> local_zbuf(static_cast<std::size_t>(BLOCK_SIZE));
        const double sum_pay = mc_call_exact_sum(p, path_seed, local_zbuf);
        const double disc = std::exp(-p.r * p.t);
        mc_price[i] = disc * sum_pay / static_cast<double>(PATHS_PER_OPTION);
        abs_err[i] = std::abs(mc_price[i] - analytical[i]);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    double max_err = 0.0;
    double mean_err = 0.0;
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        max_err = std::max(max_err, abs_err[static_cast<std::size_t>(i)]);
        mean_err += abs_err[static_cast<std::size_t>(i)];
    }
    mean_err /= static_cast<double>(NUM_OPTIONS);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes Monte Carlo\n";
    std::cout << "Options n = " << NUM_OPTIONS << ", paths per option N = " << PATHS_PER_OPTION
              << ", block_size = " << BLOCK_SIZE << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "No OpenMP -> single-threaded mode\n";
#endif
    std::cout << "Wall time: " << sec << " s\n";
    std::cout << "Mean |MC - analytical|: " << mean_err << "\n";
    std::cout << "Max  |MC - analytical|: " << max_err << "\n\n";

    const int show = std::min(5, NUM_OPTIONS);
    std::cout << "First " << show << " options:\n";
    std::cout << std::setw(4) << std::right << "i" << std::setw(12) << "S0" << std::setw(12) << "K"
              << std::setw(10) << "r" << std::setw(10) << "sigma" << std::setw(10) << "T"
              << std::setw(14) << "C_analytic" << std::setw(14) << "C_MC" << std::setw(12) << "abs_err"
              << "\n";
    std::cout << std::string(94, '-') << "\n";
    for (int i = 0; i < show; ++i) {
        const OptionParams& p = params[static_cast<std::size_t>(i)];
        std::cout << std::setw(4) << std::right << i << std::setw(12) << p.s0 << std::setw(12) << p.k
                  << std::setw(10) << p.r << std::setw(10) << p.sigma << std::setw(10) << p.t
                  << std::setw(14) << analytical[static_cast<std::size_t>(i)]
                  << std::setw(14) << mc_price[static_cast<std::size_t>(i)]
                  << std::setw(12) << abs_err[static_cast<std::size_t>(i)] << "\n";
    }

    return 0;
}
