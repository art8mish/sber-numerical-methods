/*
 * logf tests: vs std::log(double), errno/FE vs libm, ULP <= 4.
 */

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

extern "C" float logf(float x);

namespace {

constexpr double MAX_ULP = 4.0;

double ulp_error(float got, double ref) {
    if (std::isnan(static_cast<double>(got)) && std::isnan(ref)) {
        return 0.0;
    }
    const float rf = static_cast<float>(ref);
    const float n =
        std::nextafterf(rf, rf + 1e30f * (rf >= 0.f ? 1.f : -1.f));
    const double spacing =
        std::fabs(static_cast<double>(n) - static_cast<double>(rf));
    if (spacing == 0.0) {
        return 0.0;
    }
    return std::fabs(static_cast<double>(got) - ref) / spacing;
}

struct LibmRef {
    double value;
    int errno_after;
    int fe_after;
};

LibmRef reference_log(double x) {
    LibmRef r;
    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;
    volatile double xv = x;
    r.value = std::log(xv);
    r.errno_after = errno;
    r.fe_after = fetestexcept(FE_ALL_EXCEPT);
    return r;
}

bool same_fe_mask(int a, int b) {
    const int mask = FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW |
                     FE_UNDERFLOW;
    return (a & mask) == (b & mask);
}

bool test_special(float x) {
    LibmRef ref = reference_log(static_cast<double>(x));
    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;
    volatile float xv = x;
    float y = logf(xv);
    if (ref.errno_after != errno) {
        return false;
    }
    if (!same_fe_mask(ref.fe_after, fetestexcept(FE_ALL_EXCEPT))) {
        return false;
    }
    if (std::isnan(static_cast<double>(y)) && std::isnan(ref.value)) {
        return true;
    }
    if (y == static_cast<float>(ref.value)) {
        return true;
    }
    return ulp_error(y, ref.value) <= MAX_ULP;
}

bool test_point_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } xv;
    xv.u = bits;
    const float x = xv.f;
    LibmRef ref = reference_log(static_cast<double>(x));
    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;
    const float y = logf(x);
    if (ref.errno_after != errno) {
        return false;
    }
    if (!same_fe_mask(ref.fe_after, fetestexcept(FE_ALL_EXCEPT))) {
        return false;
    }
    if (std::isnan(static_cast<double>(y)) && std::isnan(ref.value)) {
        return true;
    }
    return ulp_error(y, ref.value) <= MAX_ULP;
}

bool test_interval(uint32_t lo, uint32_t hi, int n) {
    if (lo > hi || n < 2) {
        return false;
    }
    const uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    for (int i = 0; i < n; ++i) {
        const uint32_t u =
            lo + static_cast<uint32_t>((span * static_cast<uint64_t>(i)) /
                                       static_cast<uint64_t>(n - 1));
        union {
            uint32_t bits;
            float f;
        } xv;
        xv.bits = u;
        const float x = xv.f;
        if (!(x > 0.f) || !std::isfinite(x)) {
            continue;
        }
        LibmRef ref = reference_log(static_cast<double>(x));
        feclearexcept(FE_ALL_EXCEPT);
        errno = 0;
        const float y = logf(x);
        (void)y;
        if (errno != 0) {
            return false;
        }
        if (ulp_error(y, ref.value) > MAX_ULP) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const float specials[] = {
        0.f,
        std::copysign(0.f, -1.f),
        -1.f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (float x : specials) {
        if (!test_special(x)) {
            std::fputs("FAIL: special\n", stderr);
            return 1;
        }
    }
    // Explicit edge cases mentioned in review.
    const uint32_t explicit_points[] = {
        0x3F7FFFFFu,  // closest float < 1.0
        0xBF800000u,  // -1.0
        0xFF800000u,  // -inf
        0x80000001u,  // negative smallest subnormal
        0x807FFFFFu,  // negative largest subnormal
    };
    for (uint32_t u : explicit_points) {
        if (!test_point_bits(u)) {
            std::fputs("FAIL: explicit point\n", stderr);
            return 1;
        }
    }
    // Dense neighborhood near 1.0 where small absolute errors inflate in ULP.
    if (!test_interval(0x3F700000u, 0x3F8FFFFFu, 20000)) {
        std::fputs("FAIL: interval near one\n", stderr);
        return 1;
    }

    const int pts = 1000;
    if (!test_interval(1u, 0x007FFFFFu, pts)) {
        std::fputs("FAIL: interval subnormal\n", stderr);
        return 1;
    }
    const uint32_t fin_lo = 0x00800000u;
    const uint32_t fin_hi = 0x7F7FFFFFu;
    const uint64_t fspan = static_cast<uint64_t>(fin_hi - fin_lo);
    const int nbuckets = 4;
    for (int b = 0; b < nbuckets; ++b) {
        const uint32_t blo =
            fin_lo + static_cast<uint32_t>((fspan * static_cast<uint64_t>(b)) /
                                           static_cast<uint64_t>(nbuckets));
        const uint32_t bhi =
            fin_lo + static_cast<uint32_t>((fspan * static_cast<uint64_t>(b + 1)) /
                                           static_cast<uint64_t>(nbuckets));
        if (!test_interval(blo, bhi, pts)) {
            std::fputs("FAIL: interval normal\n", stderr);
            return 1;
        }
    }

    std::puts("PASS");
    return 0;
}
