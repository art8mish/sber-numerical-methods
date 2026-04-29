#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr double LN2 =
    0.693147180559945309417232121458176568075500134360255254120680009;
constexpr double SQRT2 = 1.4142135623730950488016887242096980785696718753769;

// ln(r), r in [1, 2): ln(r) = 2*atanh(u), u = (r-1)/(r+1), u in [0, 1/3].
double log_mantissa(double r) {
    const double u = (r - 1.0) / (r + 1.0);
    const double p = u * u;
    // sum_{k=0}^{10} p^k/(2k+1)
    double q = 1.0 / 21.0;
    q = 1.0 / 19.0 + p * q;
    q = 1.0 / 17.0 + p * q;
    q = 1.0 / 15.0 + p * q;
    q = 1.0 / 13.0 + p * q;
    q = 1.0 / 11.0 + p * q;
    q = 1.0 / 9.0 + p * q;
    q = 1.0 / 7.0 + p * q;
    q = 1.0 / 5.0 + p * q;
    q = 1.0 / 3.0 + p * q;
    q = 1.0 + p * q;
    return 2.0 * (u * q);
}

}  // namespace

extern "C" float logf(float x) {
    uint32_t ix = 0u;
    std::memcpy(&ix, &x, sizeof(ix));
    // 0x7FFFFFFFu = 0111 1111 1111 1111 1111 1111 1111 1111
    const uint32_t abs_ix = ix & 0x7FFFFFFFu; 

    // log(0)=-INF,
    if (abs_ix == 0u) {
        errno = ERANGE;
        feraiseexcept(FE_DIVBYZERO);
        return -std::numeric_limits<float>::infinity();
    }

    // log(negative)=NaN 
    // 0x7FFFFFFFu = 0111 1111 1111 1111 1111 1111 1111 1111
    if (ix > 0x7FFFFFFFu) {
        errno = EDOM;
        feraiseexcept(FE_INVALID);
        return std::numeric_limits<float>::quiet_NaN();
    }

    // 0x7F800000u = 0111 1111 1000 0000 0000 0000 0000 0000
    if (abs_ix >= 0x7F800000u) { 
        return x; // (abs_ix == 0x7F800000u) ? +INF : NaN
    }

    int e;
    uint32_t mx;

    // subnormal
    // 0x00800000u = 0000 0000 1000 0000 0000 0000 0000 0000
    if (abs_ix < 0x00800000u) {
        mx = abs_ix;
        e = -126;
        while ((mx & 0x00800000u) == 0u) {
            mx <<= 1;
            --e;
        }
    } else {
        e = static_cast<int>((abs_ix >> 23) & 0xFF) - 127;
        mx = 0x00800000u | (abs_ix & 0x007FFFFFu);
    }

    // 0x3F800000u = 0011 1111 1000 0000 0000 0000 0000 0000 (1.0)
    // 0x007FFFFFu = 0000 0000 0111 1111 1111 1111 1111 1111 
    const uint32_t rbits = 0x3F800000u | (mx & 0x007FFFFFu);
    float rf = 0.0f;
    std::memcpy(&rf, &rbits, sizeof(rf));
    double r = static_cast<double>(rf);
    if (r > SQRT2) {
        r *= 0.5;
        ++e;
    }
    const double lnr = log_mantissa(r);
    const double y = static_cast<double>(e) * LN2 + lnr;
    return static_cast<float>(y);
}
