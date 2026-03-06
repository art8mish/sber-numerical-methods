#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <limits>

template<typename T>
std::vector<T> generate_sample(size_t n, double mean, double stddev, unsigned seed = 777) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, stddev);
    std::vector<T> sample (n);
    for (size_t i = 0; i < n; ++i) {
        sample[i] = dist(gen);
    }
    return sample;
}

// DX = E(x^2) - (E(x))^2
template<typename T>
T fast_variance(const std::vector<T>& data) {
    if (data.empty()) 
        return 0;

    T sum = 0;
    T sum_sq = 0;
    for (T x : data) {
        sum += x;
        sum_sq += x * x;
    }
    T n = static_cast<T>(data.size());
    T mean = sum / n;
    T mean_sq = sum_sq / n;
    return mean_sq - mean * mean;
}

template<typename T>
T two_pass_variance(const std::vector<T>& data) {
    if (data.empty()) 
        return 0;

    T n = static_cast<T>(data.size());
    T sum = 0;
    for (T x : data)
        sum += x;
    T mean = sum / n;
    T sum_sq_diff = 0;
    for (T x : data) {
        T diff = x - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / n;
}

template<typename T>
T one_pass_variance(const std::vector<T>& data) {
    if (data.empty()) 
        return 0;

    T mean = data[0];
    T var = 0;
    size_t n = 1;
    size_t size = data.size();
    for (size_t i = 1; i < size; ++i) {
        T x = data[i];
        T next_n = static_cast<T>(n + 1);
        T next_mean = mean + (x - mean) / next_n;
        T next_var = var + ((x - mean) * (x - next_mean) - var) / next_n;
        mean = next_mean;
        var = next_var;
        ++n;
    }
    return var;
}


template<typename T>
void print_results(const std::vector<T>& sample, const std::string& label, double true_var) {
    std::cout << "\n=== " << label << " (n=" << sample.size() << ") ===\n";
    std::cout << std::scientific;// << std::setprecision(6);

    T fast = fast_variance(sample);
    T two_pass = two_pass_variance(sample);
    T one_pass = one_pass_variance(sample);

    std::cout << "Variance: " << true_var << "\n";
    std::cout << "Fast variance: " << fast << ",  error = " << std::fixed << std::setprecision(1) << std::abs(fast - true_var) / true_var * 100 << "%\n" << std::defaultfloat;
    std::cout << "Two-pass:      " << two_pass << ",  error = " << std::fixed << std::setprecision(1) << std::abs(two_pass - true_var) / true_var * 100 << "%\n" << std::defaultfloat;
    std::cout << "One-pass:      " << one_pass << ",  error = " << std::fixed << std::setprecision(1) << std::abs(one_pass - true_var) / true_var * 100 << "%\n" << std::defaultfloat;
}

struct Sample {
    double mean;
    double stddev;
    std::string description;
};

void run_experiment(size_t n, double mean, double stddev) {
    std::vector<double> data_d = generate_sample<double>(n, mean, stddev);
    std::vector<float> data_f (n);
    for (size_t i = 0; i < n; ++i) 
        data_f[i] = static_cast<float>(data_d[i]);

    double err = stddev * stddev;
    auto print_row = [&](std::string type, std::string method, double result) {
        double rel_err = std::abs(result - err) / err;
        std::cout << std::left << std::setw(8) << type 
                  << std::setw(12) << method 
                  << std::scientific << std::setprecision(6) << result 
                  << " | Err: " << std::fixed << std::setprecision(1) << rel_err * 100 << "%" << std::defaultfloat << std::endl;
    };

    std::cout << "\n-------- Test: Mean=" << mean << ", StdDev=" << stddev << ") --------" << std::endl;
    
    print_row("Double", "Fast", fast_variance(data_d));
    print_row("Double", "Two-Pass", two_pass_variance(data_d));
    print_row("Double", "One-Pass", one_pass_variance(data_d));
    std::cout << "------------------------------------------------" << std::endl;
    print_row("Float",  "Fast", fast_variance(data_f));
    print_row("Float",  "Two-Pass", two_pass_variance(data_f));
    print_row("Float",  "One-Pass", one_pass_variance(data_f));
    std::cout << "------------------------------------------------" << std::endl;
}

int main() {
    run_experiment(1000, 1.0, 1.0);
    run_experiment(1000, 10.0, 0.1);
    run_experiment(1000, 100.0, 0.01);
    return 0;
}