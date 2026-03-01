#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <limits>

template<typename T>
std::vector<T> generate_sample(size_t n, double mean, double stddev, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, stddev);
    std::vector<T> sample;
    sample.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        sample.push_back(dist(gen));
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
    std::cout << "Fast variance: " << fast << ",  error = " << std::abs(fast - true_var) / true_var << "\n";
    std::cout << "Two-pass:      " << two_pass << ",  error = " << std::abs(two_pass - true_var) / true_var << "\n";
    std::cout << "One-pass:      " << one_pass << ",  error = " << std::abs(one_pass - true_var) / true_var << "\n";
}

struct Sample {
    double mean;
    double stddev;
    std::string description;
};


int main() {
    Sample samples[] = {
        {1.0, 1.0, "mean=1, stddev=1"},
        {10.0, 0.1, "mean=10, stddev=0.1"},
        {100.0, 0.01, "mean=100, stddev=0.01"}
    };
    const int N = 1000;

    std::cout << "\nExperiments for float:\n";
    for (const auto& s : samples) {
        std::vector<float> sample_float = generate_sample<float>(N, s.mean, s.stddev);
        print_results<float>(sample_float, s.description, s.stddev);
    }

    std::cout << "\nExperiments for double:\n";
    for (const auto& s : samples) {
        std::vector<double> sample_double = generate_sample<double>(N, s.mean, s.stddev);
        print_results<double>(sample_double, s.description, s.stddev);
    }

    return 0;
}