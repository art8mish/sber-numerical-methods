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

void matmul_naive(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n) {
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

void matmul_blocked(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n, size_t block) {
    std::fill(c.begin(), c.end(), 0.0f);
    Matrix bt(n * k);
    transpose(b, bt, k, n);

    for (size_t jj = 0; jj < n; jj += block) {
        for (size_t ii = 0; ii < m; ii += block) {
            for (size_t kk = 0; kk < k; kk += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        Value sum = c[idx(i, j, n)];
                        for (size_t p = kk; p < k_end; ++p) {
                            sum += a[idx(i, p, k)] * bt[idx(j, p, k)];
                        }
                        c[idx(i, j, n)] = sum;
                    }
                }
            }
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
float sum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void dot_avx2_fma(const Matrix& a, const Matrix& bt, size_t kcols, size_t i, size_t j, size_t k0,
                                 size_t k1, Value* acc) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    size_t p = k0;
    for (; p + 16 <= k1; p += 16) {
        const __m256 va0 = _mm256_loadu_ps(&a[idx(i, p, kcols)]);
        const __m256 va1 = _mm256_loadu_ps(&a[idx(i, p + 8, kcols)]);
        const __m256 vb0 = _mm256_loadu_ps(&bt[idx(j, p, kcols)]);
        const __m256 vb1 = _mm256_loadu_ps(&bt[idx(j, p + 8, kcols)]);
        s0 = _mm256_fmadd_ps(va0, vb0, s0);
        s1 = _mm256_fmadd_ps(va1, vb1, s1);
    }
    __m256 sumv = _mm256_add_ps(s0, s1);
    for (; p + 8 <= k1; p += 8) {
        const __m256 va = _mm256_loadu_ps(&a[idx(i, p, kcols)]);
        const __m256 vb = _mm256_loadu_ps(&bt[idx(j, p, kcols)]);
        sumv = _mm256_fmadd_ps(va, vb, sumv);
    }
    float part = sum_avx2(sumv);
    for (; p < k1; ++p)
        part += a[idx(i, p, kcols)] * bt[idx(j, p, kcols)];
    *acc += part;
}

void matmul_blocked_simd(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n, size_t block) {
    std::fill(c.begin(), c.end(), 0.0f);
    Matrix bt(n * k);
    transpose(b, bt, k, n);

    for (size_t jj = 0; jj < n; jj += block) {
        for (size_t ii = 0; ii < m; ii += block) {
            for (size_t kk = 0; kk < k; kk += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        Value sum = c[idx(i, j, n)];
                        dot_avx2_fma(a, bt, k, i, j, kk, k_end, &sum);
                        c[idx(i, j, n)] = sum;
                    }
                }
            }
        }
    }
}
#endif

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
        {32, 64, 16, "32x64*64x16"},
        {64, 64, 64, "64x64*64x64"},
        {128, 64, 64, "128x64*64x64"},
        {128, 128, 128, "128x128^2"},
        // {1024, 1024, 1024, "1024^3"},
    };

    std::cout << "fp32 GEMM, block=" << block << "\n";

#if defined(__AVX2__) && defined(__FMA__)
    std::cout << "case | naive @ | blocked @ | simd @ | n/blk | n/simd |\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (const Shape& s : shapes) {
        Matrix a = random_matrix(s.m, s.k, 777u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix b = random_matrix(s.k, s.n, 999u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix c0(s.m * s.n), c1(s.m * s.n), c2(s.m * s.n);

        const uint64_t t0 = benchmark_cycles([&]() { matmul_naive(a, b, c0, s.m, s.k, s.n); });
        const uint64_t t1 = benchmark_cycles([&]() { matmul_blocked(a, b, c1, s.m, s.k, s.n, block); });
        const uint64_t t2 = benchmark_cycles([&]() { matmul_blocked_simd(a, b, c2, s.m, s.k, s.n, block); });

        const double sb = t1 > 0 ? static_cast<double>(t0) / static_cast<double>(t1) : 0.0;
        const double ss = t2 > 0 ? static_cast<double>(t0) / static_cast<double>(t2) : 0.0;

        std::cout << std::setw(14) << s.name << " | " << std::setw(9) << t0 << " | " << std::setw(11) << t1 << " | "
                  << std::setw(8) << t2 << " | " << std::setw(6) << sb << " | " << std::setw(7) << ss << " | " << "\n";
    }
#else
    std::cout << "case | naive @ | blocked @ | speedup\n";
    std::cout << "-------------------------------------------------------\n";

    for (const Shape& s : shapes) {
        Matrix a = random_matrix(s.m, s.k, 777u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix b = random_matrix(s.k, s.n, 999u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix c0(s.m * s.n), c1(s.m * s.n);

        const uint64_t t0 = benchmark_cycles([&]() { matmul_naive(a, b, c0, s.m, s.k, s.n); });
        const uint64_t t1 = benchmark_cycles([&]() { matmul_blocked(a, b, c1, s.m, s.k, s.n, block); });
        const double sp = t1 > 0 ? static_cast<double>(t0) / static_cast<double>(t1) : 0.0;

        std::cout << std::setw(14) << s.name << " | " << std::setw(9) << t0 << " | " << std::setw(11) << t1 << " | "
                  << std::setw(7) << sp << " | " << "\n";
    }
#endif

    return 0;
}
