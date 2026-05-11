
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <limits>

__m256 logf8_avx2(__m256 x);

namespace {

constexpr double MAX_ULP = 4.0;

double ulp_error(float got, double ref) {
    const float rf = static_cast<float>(ref);
    const float n = std::nextafterf(rf, rf + 1e30f * (rf >= 0.f ? 1.f : -1.f));
    const double spacing = std::fabs(static_cast<double>(n) - static_cast<double>(rf));
    if (spacing == 0.0) {
        return 0.0;
    }
    return std::fabs(static_cast<double>(got) - ref) / spacing;
}

__attribute__((target("avx2,fma"))) void call_logf8(const float *in, float *out) {
    __m256 v = _mm256_loadu_ps(in);
    __m256 r = logf8_avx2(v);
    _mm256_storeu_ps(out, r);
}

struct Stats {
    double max_ulp = 0.0;
    uint64_t checked = 0;
    uint64_t bad = 0;
    uint32_t worst_bits = 0;
    float worst_x = 0.f;
    float worst_got = 0.f;
    double worst_ref = 0.0;
};

void update_stats(Stats &s, float x, uint32_t bits, float got) {
    const double ref = std::log(static_cast<double>(x));
    const double e = ulp_error(got, ref);
    if (e > s.max_ulp) {
        s.max_ulp = e;
        s.worst_bits = bits;
        s.worst_x = x;
        s.worst_got = got;
        s.worst_ref = ref;
    }
    if (e > MAX_ULP) {
        ++s.bad;
    }
    ++s.checked;
}

void run_batch(const uint32_t bits[8], Stats &s) {
    alignas(32) float in[8];
    alignas(32) float out[8];
    for (int i = 0; i < 8; ++i) {
        in[i] = std::bit_cast<float>(bits[i]);
    }
    call_logf8(in, out);
    for (int i = 0; i < 8; ++i) {
        update_stats(s, in[i], bits[i], out[i]);
    }
}

bool valid_normalized(uint32_t u) {
    const uint32_t biased = (u >> 23) & 0xFFu;
    return biased >= 1u && biased <= 254u && (u & 0x80000000u) == 0u;
}

void test_interval(uint32_t lo, uint32_t hi, uint64_t n, Stats &s) {
    if (lo > hi || n < 8) {
        return;
    }
    const uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    uint32_t batch[8];
    int k = 0;
    uint32_t last_valid = lo;
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t u = lo + static_cast<uint32_t>((span * i) / (n - 1));
        if (!valid_normalized(u)) {
            continue;
        }
        last_valid = u;
        batch[k++] = u;
        if (k == 8) {
            run_batch(batch, s);
            k = 0;
        }
    }
    if (k > 0) {
        for (int j = k; j < 8; ++j) {
            batch[j] = last_valid;
        }
        run_batch(batch, s);
    }
}

void test_explicit(const float *xs, int n, Stats &s) {
    int i = 0;
    while (i < n) {
        uint32_t batch[8];
        int k = 0;
        while (k < 8 && i < n) {
            const uint32_t u = std::bit_cast<uint32_t>(xs[i++]);
            if (!valid_normalized(u)) {
                continue;
            }
            batch[k++] = u;
        }
        if (k == 0) {
            break;
        }
        for (int j = k; j < 8; ++j) {
            batch[j] = batch[0];
        }
        run_batch(batch, s);
    }
}

} // namespace

int main() {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::cout << "SKIPPED: CPU does not support AVX2+FMA\n";
        return 0;
    }

    Stats s;

    const float interesting[] = {
        1.0f,
        std::nextafterf(1.0f, 2.0f),
        std::nextafterf(1.0f, 0.0f),
        2.0f,
        0.5f,
        1.4142135f,
        1.4142136f,
        std::nextafterf(1.4142135623730951f, 2.0f),
        std::nextafterf(1.4142135623730951f, 1.0f),
        2.7182817f,
        10.0f,
        100.0f,
        1e10f,
        1e-10f,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
    };
    test_explicit(interesting, static_cast<int>(sizeof(interesting) / sizeof(interesting[0])), s);

    test_interval(0x3F700000u, 0x3F8FFFFFu, 20000, s);
    test_interval(0x3FB00000u, 0x3FB80000u, 5000, s);

    const uint32_t fin_lo = 0x00800000u;
    const uint32_t fin_hi = 0x7F7FFFFFu;
    const uint64_t fspan = static_cast<uint64_t>(fin_hi - fin_lo);
    const int nbuckets = 4;
    const uint64_t pts_per = 5000;
    for (int b = 0; b < nbuckets; ++b) {
        const uint32_t blo =
            fin_lo + static_cast<uint32_t>((fspan * static_cast<uint64_t>(b)) / nbuckets);
        const uint32_t bhi =
            fin_lo + static_cast<uint32_t>((fspan * static_cast<uint64_t>(b + 1)) / nbuckets);
        test_interval(blo, bhi, pts_per, s);
    }

    std::cout << std::format("checked={} bad={} max_ulp={:.4f} worst_bits=0x{:08X} "
                             "worst_x={:.9g} got={:.9g} ref={:.17g}\n",
                             s.checked, s.bad, s.max_ulp, s.worst_bits, s.worst_x,
                             s.worst_got, s.worst_ref);

    if (s.bad > 0 || s.max_ulp > MAX_ULP) {
        std::cout << "FAIL\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
