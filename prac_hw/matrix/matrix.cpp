#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

using Value = float;
using Matrix = std::vector<Value>;

struct Shape {
    size_t m;
    size_t k;
    size_t n;
    std::string name;
};

inline size_t idx(size_t i, size_t j, size_t cols) {
    return i * cols + j;
}

Matrix random_matrix(size_t rows, size_t cols, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Value> dist(-1.0f, 1.0f);
    Matrix m(rows * cols);
    for (Value& v : m) {
        v = dist(gen);
    }
    return m;
}

void multiply_naive(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Value sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += a[idx(i, p, k)] * b[idx(p, j, n)];
            }
            c[idx(i, j, n)] = sum;
        }
    }
}

void transpose(const Matrix& src, Matrix& dst, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[idx(j, i, rows)] = src[idx(i, j, cols)];
        }
    }
}

void multiply_blocked_transposed(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n, size_t block) {
    std::fill(c.begin(), c.end(), 0.0f);
    Matrix bt(n * k);
    transpose(b, bt, k, n);

    for (size_t ii = 0; ii < m; ii += block) {
        for (size_t jj = 0; jj < n; jj += block) {
            for (size_t kk = 0; kk < k; kk += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        Value sum = c[idx(i, j, n)];
                        for (size_t p = kk; p < k_end; ++p)
                            sum += a[idx(i, p, k)] * bt[idx(j, p, k)];
                        c[idx(i, j, n)] = sum;
                    }
                }
            }
        }
    }
}

Value max_abs_diff(const Matrix& x, const Matrix& y) {
    Value diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        diff = std::max(diff, std::abs(x[i] - y[i]));
    }
    return diff;
}

#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t read_tsc_start() {
    unsigned aux = 0;
    (void)aux;
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

template <typename Func>
uint64_t benchmark_cycles(Func f, int repeat = 5) {
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < repeat; ++r) {
#if defined(__x86_64__) || defined(__i386__)
        const uint64_t t1 = read_tsc_start();
        f();
        const uint64_t t2 = read_tsc_end();
        best = std::min(best, t2 - t1);
#else
        f();
        best = 0;
#endif
    }
    return best;
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    const size_t block = 16;
    const std::vector<Shape> shapes = {
        {32, 64, 16, "A32x64 * B64x16"},
        {64, 64, 64, "A64x64 * B64x64"},
        {128, 64, 64, "A128x64 * B64x64"},
        {128, 128, 128, "A128x128 * B128x128"},
    };

    std::cout << "Matrix benchmark (fp32), block=" << block << "\n";
#if defined(__x86_64__) || defined(__i386__)
    std::cout << "Case                 | Naive cycles | Blocked cycles | Speedup | Max abs diff\n";
    std::cout << "-------------------------------------------------------------------------------\n";
#else
    std::cout << "Case                 | Naive cycles | Blocked cycles | Speedup | Max abs diff\n";
    std::cout << "(warning: cycle counter is unavailable on this architecture)\n";
#endif

    for (const Shape& s : shapes) {
        Matrix a = random_matrix(s.m, s.k, 777u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix b = random_matrix(s.k, s.n, 999u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix c_naive(s.m * s.n);
        Matrix c_block(s.m * s.n);

        const uint64_t naive_cycles =
            benchmark_cycles([&]() { multiply_naive(a, b, c_naive, s.m, s.k, s.n); }, 5);
        const uint64_t block_cycles = benchmark_cycles(
            [&]() { multiply_blocked_transposed(a, b, c_block, s.m, s.k, s.n, block); }, 5);
        const double speedup = block_cycles > 0
                                   ? static_cast<double>(naive_cycles) / static_cast<double>(block_cycles)
                                   : 0.0;
        const Value diff = max_abs_diff(c_naive, c_block);

        std::cout << std::setw(20) << s.name << " | " << std::setw(12) << naive_cycles << " | "
                  << std::setw(14) << block_cycles << " | " << std::setw(7) << speedup << " | " << diff
                  << "\n";
    }

    return 0;
}
