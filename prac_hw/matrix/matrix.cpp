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
    for (size_t jj = 0; jj < n; jj += block) {
        for (size_t kk = 0; kk < k; kk += block) {
            for (size_t ii = 0; ii < m; ii += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t p = kk; p < k_end; ++p) {
                        const Value aip = a[idx(i, p, k)];
                        for (size_t j = jj; j < j_end; ++j) {
                            c[idx(i, j, n)] += aip * b[idx(p, j, n)];
                        }
                    }
                }
            }
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
void matmul_blocked_simd(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n, size_t block) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (size_t jj = 0; jj < n; jj += block) {
        for (size_t kk = 0; kk < k; kk += block) {
            for (size_t ii = 0; ii < m; ii += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);

                size_t i = ii;
                for (; i + 1 < i_end; i += 2) {
                    size_t j = jj;
                    for (; j + 16 <= j_end; j += 16) {
                        __m256 vc00 = _mm256_loadu_ps(&c[idx(i + 0, j + 0, n)]);
                        __m256 vc01 = _mm256_loadu_ps(&c[idx(i + 0, j + 8, n)]);
                        __m256 vc10 = _mm256_loadu_ps(&c[idx(i + 1, j + 0, n)]);
                        __m256 vc11 = _mm256_loadu_ps(&c[idx(i + 1, j + 8, n)]);
                        for (size_t p = kk; p < k_end; ++p) {
                            const __m256 vb0 = _mm256_loadu_ps(&b[idx(p, j + 0, n)]);
                            const __m256 vb1 = _mm256_loadu_ps(&b[idx(p, j + 8, n)]);
                            const __m256 va0 = _mm256_set1_ps(a[idx(i + 0, p, k)]);
                            const __m256 va1 = _mm256_set1_ps(a[idx(i + 1, p, k)]);
                            vc00 = _mm256_fmadd_ps(va0, vb0, vc00);
                            vc01 = _mm256_fmadd_ps(va0, vb1, vc01);
                            vc10 = _mm256_fmadd_ps(va1, vb0, vc10);
                            vc11 = _mm256_fmadd_ps(va1, vb1, vc11);
                        }
                        _mm256_storeu_ps(&c[idx(i + 0, j + 0, n)], vc00);
                        _mm256_storeu_ps(&c[idx(i + 0, j + 8, n)], vc01);
                        _mm256_storeu_ps(&c[idx(i + 1, j + 0, n)], vc10);
                        _mm256_storeu_ps(&c[idx(i + 1, j + 8, n)], vc11);
                    }
                    for (; j + 8 <= j_end; j += 8) {
                        __m256 vc0 = _mm256_loadu_ps(&c[idx(i + 0, j, n)]);
                        __m256 vc1 = _mm256_loadu_ps(&c[idx(i + 1, j, n)]);
                        for (size_t p = kk; p < k_end; ++p) {
                            const __m256 vb = _mm256_loadu_ps(&b[idx(p, j, n)]);
                            const __m256 va0 = _mm256_set1_ps(a[idx(i + 0, p, k)]);
                            const __m256 va1 = _mm256_set1_ps(a[idx(i + 1, p, k)]);
                            vc0 = _mm256_fmadd_ps(va0, vb, vc0);
                            vc1 = _mm256_fmadd_ps(va1, vb, vc1);
                        }
                        _mm256_storeu_ps(&c[idx(i + 0, j, n)], vc0);
                        _mm256_storeu_ps(&c[idx(i + 1, j, n)], vc1);
                    }
                    for (; j < j_end; ++j) {
                        Value sum0 = c[idx(i + 0, j, n)];
                        Value sum1 = c[idx(i + 1, j, n)];
                        for (size_t p = kk; p < k_end; ++p) {
                            const Value bpj = b[idx(p, j, n)];
                            sum0 += a[idx(i + 0, p, k)] * bpj;
                            sum1 += a[idx(i + 1, p, k)] * bpj;
                        }
                        c[idx(i + 0, j, n)] = sum0;
                        c[idx(i + 1, j, n)] = sum1;
                    }
                }

                for (; i < i_end; ++i) {
                    size_t j = jj;
                    for (; j + 16 <= j_end; j += 16) {
                        __m256 vc0 = _mm256_loadu_ps(&c[idx(i, j + 0, n)]);
                        __m256 vc1 = _mm256_loadu_ps(&c[idx(i, j + 8, n)]);
                        for (size_t p = kk; p < k_end; ++p) {
                            const __m256 va = _mm256_set1_ps(a[idx(i, p, k)]);
                            vc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j + 0, n)]), vc0);
                            vc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j + 8, n)]), vc1);
                        }
                        _mm256_storeu_ps(&c[idx(i, j + 0, n)], vc0);
                        _mm256_storeu_ps(&c[idx(i, j + 8, n)], vc1);
                    }
                    for (; j + 8 <= j_end; j += 8) {
                        __m256 vc = _mm256_loadu_ps(&c[idx(i, j, n)]);
                        for (size_t p = kk; p < k_end; ++p) {
                            const __m256 va = _mm256_set1_ps(a[idx(i, p, k)]);
                            vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j, n)]), vc);
                        }
                        _mm256_storeu_ps(&c[idx(i, j, n)], vc);
                    }
                    for (; j < j_end; ++j) {
                        Value sum = c[idx(i, j, n)];
                        for (size_t p = kk; p < k_end; ++p) {
                            sum += a[idx(i, p, k)] * b[idx(p, j, n)];
                        }
                        c[idx(i, j, n)] = sum;
                    }
                }
            }
        }
    }
}

// Same blocking and loop order as matmul_blocked: i, then p, then j (SIMD over j; load/fma/store C each p).
void matmul_blocked_simd_ipj(const Matrix& a, const Matrix& b, Matrix& c, size_t m, size_t k, size_t n, size_t block) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (size_t jj = 0; jj < n; jj += block) {
        for (size_t kk = 0; kk < k; kk += block) {
            for (size_t ii = 0; ii < m; ii += block) {
                const size_t i_end = std::min(ii + block, m);
                const size_t j_end = std::min(jj + block, n);
                const size_t k_end = std::min(kk + block, k);
                size_t i = ii;
                for (; i + 1 < i_end; i += 2) {
                    for (size_t p = kk; p < k_end; ++p) {
                        const __m256 va0 = _mm256_set1_ps(a[idx(i + 0, p, k)]);
                        const __m256 va1 = _mm256_set1_ps(a[idx(i + 1, p, k)]);
                        size_t j = jj;
                        for (; j + 16 <= j_end; j += 16) {
                            __m256 vc00 = _mm256_loadu_ps(&c[idx(i + 0, j + 0, n)]);
                            __m256 vc01 = _mm256_loadu_ps(&c[idx(i + 0, j + 8, n)]);
                            __m256 vc10 = _mm256_loadu_ps(&c[idx(i + 1, j + 0, n)]);
                            __m256 vc11 = _mm256_loadu_ps(&c[idx(i + 1, j + 8, n)]);
                            const __m256 vb0 = _mm256_loadu_ps(&b[idx(p, j + 0, n)]);
                            const __m256 vb1 = _mm256_loadu_ps(&b[idx(p, j + 8, n)]);
                            vc00 = _mm256_fmadd_ps(va0, vb0, vc00);
                            vc01 = _mm256_fmadd_ps(va0, vb1, vc01);
                            vc10 = _mm256_fmadd_ps(va1, vb0, vc10);
                            vc11 = _mm256_fmadd_ps(va1, vb1, vc11);
                            _mm256_storeu_ps(&c[idx(i + 0, j + 0, n)], vc00);
                            _mm256_storeu_ps(&c[idx(i + 0, j + 8, n)], vc01);
                            _mm256_storeu_ps(&c[idx(i + 1, j + 0, n)], vc10);
                            _mm256_storeu_ps(&c[idx(i + 1, j + 8, n)], vc11);
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vc0 = _mm256_loadu_ps(&c[idx(i + 0, j, n)]);
                            __m256 vc1 = _mm256_loadu_ps(&c[idx(i + 1, j, n)]);
                            const __m256 vb = _mm256_loadu_ps(&b[idx(p, j, n)]);
                            vc0 = _mm256_fmadd_ps(va0, vb, vc0);
                            vc1 = _mm256_fmadd_ps(va1, vb, vc1);
                            _mm256_storeu_ps(&c[idx(i + 0, j, n)], vc0);
                            _mm256_storeu_ps(&c[idx(i + 1, j, n)], vc1);
                        }
                        for (; j < j_end; ++j) {
                            const Value bpj = b[idx(p, j, n)];
                            c[idx(i + 0, j, n)] += a[idx(i + 0, p, k)] * bpj;
                            c[idx(i + 1, j, n)] += a[idx(i + 1, p, k)] * bpj;
                        }
                    }
                }
                for (; i < i_end; ++i) {
                    for (size_t p = kk; p < k_end; ++p) {
                        const __m256 va = _mm256_set1_ps(a[idx(i, p, k)]);
                        size_t j = jj;
                        for (; j + 16 <= j_end; j += 16) {
                            __m256 vc0 = _mm256_loadu_ps(&c[idx(i, j + 0, n)]);
                            __m256 vc1 = _mm256_loadu_ps(&c[idx(i, j + 8, n)]);
                            vc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j + 0, n)]), vc0);
                            vc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j + 8, n)]), vc1);
                            _mm256_storeu_ps(&c[idx(i, j + 0, n)], vc0);
                            _mm256_storeu_ps(&c[idx(i, j + 8, n)], vc1);
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vc = _mm256_loadu_ps(&c[idx(i, j, n)]);
                            vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(&b[idx(p, j, n)]), vc);
                            _mm256_storeu_ps(&c[idx(i, j, n)], vc);
                        }
                        for (; j < j_end; ++j) {
                            c[idx(i, j, n)] += a[idx(i, p, k)] * b[idx(p, j, n)];
                        }
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
        {32, 64, 16, "32x64@64x16"},
        {64, 64, 64, "64x64@64x64"},
        {128, 64, 64, "128x64@64x64"},
        {128, 128, 128, "128x128@128x128"},
        // {1024, 1024, 1024, "1024x1024^2"},
    };

    std::cout << "fp32 GEMM, block=" << block << "\n";

#if defined(__AVX2__) && defined(__FMA__)
    std::cout << std::setw(15) << "case" << " | " << std::setw(9) << "naive @" << " | " << std::setw(11) << "blocked @" << " | "  << std::setw(11) << "simd_ipj @" << " | " << std::setw(8) << "simd @" << " | " << std::setw(6) << "n/blk" << " | " << std::setw(9) << "n/simd_ipj"  << " | " << std::setw(7) << "n/simd" << " | " << "\n";
    std::cout << std::string(105, '-') << "\n";

    for (const Shape& s : shapes) {
        Matrix a = random_matrix(s.m, s.k, 777u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix b = random_matrix(s.k, s.n, 999u + static_cast<unsigned>(s.m + s.k + s.n));
        Matrix c0(s.m * s.n), c1(s.m * s.n), c2(s.m * s.n), c3(s.m * s.n);

        const uint64_t t0 = benchmark_cycles([&]() { matmul_naive(a, b, c0, s.m, s.k, s.n); });
        const uint64_t t1 = benchmark_cycles([&]() { matmul_blocked(a, b, c1, s.m, s.k, s.n, block); });
        const uint64_t t2 = benchmark_cycles([&]() { matmul_blocked_simd(a, b, c2, s.m, s.k, s.n, block); });
        const uint64_t t3 = benchmark_cycles([&]() { matmul_blocked_simd_ipj(a, b, c3, s.m, s.k, s.n, block); });

        const double sb = t1 > 0 ? static_cast<double>(t0) / static_cast<double>(t1) : 0.0;
        const double ss = t2 > 0 ? static_cast<double>(t0) / static_cast<double>(t2) : 0.0;
        const double sipj = t3 > 0 ? static_cast<double>(t0) / static_cast<double>(t3) : 0.0;

        std::cout << std::setw(15) << s.name << " | " << std::setw(9) << t0 << " | " << std::setw(11) << t1 << " | "
            << std::setw(11) << t3 << " | " << std::setw(8) << t2 << " | " << std::setw(6) << sb << " | " << std::setw(9) << sipj << " | " << std::setw(7) << ss << " | "  << "\n";
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
