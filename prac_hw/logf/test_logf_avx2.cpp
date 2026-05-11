// Тест logf8_avx2: собирается без -mavx2 (см. Makefile), AVX2-путь обёрнут
// в __attribute__((target("avx2,fma"))) и вызывается только после CPUID-проверки.

#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <span>

__m256 logf8_avx2(__m256 x);

namespace {

constexpr double MAX_ULP = 4.0;
constexpr int VEC_W = 8;

double ulp_error(float got, double ref) {
    const float rf = static_cast<float>(ref);
    const float n = std::nextafterf(rf, rf + 1e30f * (rf >= 0.f ? 1.f : -1.f));
    const double spacing = std::fabs(static_cast<double>(n) - static_cast<double>(rf));
    return spacing == 0.0 ? 0.0 : std::fabs(static_cast<double>(got) - ref) / spacing;
}

bool valid_normalized(uint32_t u) {
    const uint32_t biased = (u >> 23) & 0xFFu;
    return biased >= 1u && biased <= 254u && (u & 0x80000000u) == 0u;
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

__attribute__((target("avx2,fma"))) void call_logf8(const float *in, float *out) {
    _mm256_storeu_ps(out, logf8_avx2(_mm256_loadu_ps(in)));
}

// Буферизует валидные точки до батча в VEC_W элементов и сравнивает с std::log.
class BatchRunner {
public:
    explicit BatchRunner(Stats &s) : stats_(s) {}

    void add(uint32_t bits) {
        if (!valid_normalized(bits))
            return;
        buf_[k_++] = bits;
        last_ = bits;
        if (k_ == VEC_W)
            flush_full();
    }

    void flush() {
        if (k_ == 0)
            return;
        for (int j = k_; j < VEC_W; ++j)
            buf_[j] = last_;
        flush_full();
    }

private:
    void flush_full() {
        alignas(32) float in[VEC_W];
        alignas(32) float out[VEC_W];
        for (int i = 0; i < VEC_W; ++i)
            in[i] = std::bit_cast<float>(buf_[i]);
        call_logf8(in, out);
        for (int i = 0; i < VEC_W; ++i) {
            const float x = in[i];
            const float y = out[i];
            const double ref = std::log(static_cast<double>(x));
            const double e = ulp_error(y, ref);
            if (e > stats_.max_ulp) {
                stats_.max_ulp = e;
                stats_.worst_bits = buf_[i];
                stats_.worst_x = x;
                stats_.worst_got = y;
                stats_.worst_ref = ref;
            }
            if (e > MAX_ULP)
                ++stats_.bad;
            ++stats_.checked;
        }
        k_ = 0;
    }

    Stats &stats_;
    uint32_t buf_[VEC_W]{};
    uint32_t last_ = 0;
    int k_ = 0;
};

void sweep_interval(BatchRunner &r, uint32_t lo, uint32_t hi, uint64_t n) {
    if (lo > hi || n < 2)
        return;
    const uint64_t span = static_cast<uint64_t>(hi) - lo;
    for (uint64_t i = 0; i < n; ++i)
        r.add(lo + static_cast<uint32_t>((span * i) / (n - 1)));
}

void sweep_points(BatchRunner &r, std::span<const float> xs) {
    for (float x : xs)
        r.add(std::bit_cast<uint32_t>(x));
}

} // namespace

int main() {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::cout << "SKIPPED: CPU does not support AVX2+FMA\n";
        return 0;
    }

    Stats s;
    BatchRunner r{s};

    static const float interesting[] = {
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
    sweep_points(r, interesting);

    sweep_interval(r, 0x3F700000u, 0x3F8FFFFFu, 20000);
    sweep_interval(r, 0x3FB00000u, 0x3FB80000u, 5000);

    constexpr uint32_t fin_lo = 0x00800000u;
    constexpr uint32_t fin_hi = 0x7F7FFFFFu;
    constexpr uint64_t fspan = fin_hi - fin_lo;
    constexpr int nbuckets = 4;
    constexpr uint64_t pts_per = 5000;
    for (int b = 0; b < nbuckets; ++b) {
        const uint32_t blo = fin_lo + static_cast<uint32_t>((fspan * b) / nbuckets);
        const uint32_t bhi = fin_lo + static_cast<uint32_t>((fspan * (b + 1)) / nbuckets);
        sweep_interval(r, blo, bhi, pts_per);
    }

    r.flush();

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
