#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

extern "C" float logf(float x);

namespace {

constexpr double MAX_ULP = 4.0;
constexpr int FE_MASK = FE_DIVBYZERO | FE_INEXACT | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW;

double ulp_error(float got, double ref) {
    if (std::isnan(static_cast<double>(got)) && std::isnan(ref))
        return 0.0;
    const float rf = static_cast<float>(ref);
    const float n = std::nextafterf(rf, rf + 1e30f * (rf >= 0.f ? 1.f : -1.f));
    const double spacing = std::fabs(static_cast<double>(n) - static_cast<double>(rf));
    return spacing == 0.0 ? 0.0 : std::fabs(static_cast<double>(got) - ref) / spacing;
}

struct Probe {
    double value;
    int err;
    int fe;
};

Probe probe_logf(float x) {
    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;
    volatile float xv = x;
    const float y = logf(xv);
    return {static_cast<double>(y), errno, fetestexcept(FE_ALL_EXCEPT)};
}

Probe probe_ref(double x) {
    feclearexcept(FE_ALL_EXCEPT);
    errno = 0;
    volatile double xv = x;
    return {std::log(xv), errno, fetestexcept(FE_ALL_EXCEPT)};
}

// Особые точки: errno и FE-маска должны совпадать с libm, плюс ULP <= MAX_ULP.
bool check_libm_match(uint32_t bits) {
    const float x = std::bit_cast<float>(bits);
    const Probe ref = probe_ref(static_cast<double>(x));
    const Probe got = probe_logf(x);
    if (got.err != ref.err)
        return false;
    if ((got.fe & FE_MASK) != (ref.fe & FE_MASK))
        return false;
    if (std::isnan(got.value) && std::isnan(ref.value))
        return true;
    const float gotf = static_cast<float>(got.value);
    if (gotf == static_cast<float>(ref.value))
        return true;
    return ulp_error(gotf, ref.value) <= MAX_ULP;
}

// Прогон по интервалу битов: только errno == 0 и ULP <= MAX_ULP.
bool check_interval(uint32_t lo, uint32_t hi, int n) {
    if (lo > hi || n < 2)
        return false;
    const uint64_t span = static_cast<uint64_t>(hi) - lo;
    for (int i = 0; i < n; ++i) {
        const uint32_t u = lo + static_cast<uint32_t>((span * static_cast<uint64_t>(i)) / (n - 1));
        const float x = std::bit_cast<float>(u);
        if (!(x > 0.f) || !std::isfinite(x))
            continue;
        const Probe got = probe_logf(x);
        if (got.err != 0)
            return false;
        const double ref = std::log(static_cast<double>(x));
        if (ulp_error(static_cast<float>(got.value), ref) > MAX_ULP)
            return false;
    }
    return true;
}

void fail(const char *what) {
    std::cerr << "FAIL: " << what << "\n";
}

} // namespace

int main() {
    constexpr uint32_t specials[] = {
        0x00000000u, // +0
        0x80000000u, // -0
        0xBF800000u, // -1
        0x7F800000u, // +inf
        0xFF800000u, // -inf
        0x7FC00000u, // qNaN
        0x3F7FFFFFu, // closest float < 1.0
        0x80000001u, // smallest negative subnormal
        0x807FFFFFu, // largest negative subnormal
    };
    for (uint32_t u : specials) {
        if (!check_libm_match(u)) {
            fail("special");
            return 1;
        }
    }

    if (!check_interval(0x3F700000u, 0x3F8FFFFFu, 20000)) {
        fail("interval near one");
        return 1;
    }
    if (!check_interval(1u, 0x007FFFFFu, 1000)) {
        fail("interval subnormal");
        return 1;
    }

    constexpr uint32_t fin_lo = 0x00800000u;
    constexpr uint32_t fin_hi = 0x7F7FFFFFu;
    constexpr uint64_t fspan = fin_hi - fin_lo;
    for (int b = 0; b < 4; ++b) {
        const uint32_t blo = fin_lo + static_cast<uint32_t>((fspan * b) / 4);
        const uint32_t bhi = fin_lo + static_cast<uint32_t>((fspan * (b + 1)) / 4);
        if (!check_interval(blo, bhi, 1000)) {
            fail("interval normal");
            return 1;
        }
    }

    std::cout << "PASS\n";
    return 0;
}
