#include <immintrin.h>

namespace {

constexpr double LN2 = 0.693147180559945309417232121458176568075500134360255254120680009;

inline __m256d log_mantissa_pd(__m256d r) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d u = _mm256_div_pd(_mm256_sub_pd(r, one), _mm256_add_pd(r, one));
    const __m256d p = _mm256_mul_pd(u, u);
    __m256d q = _mm256_set1_pd(1.0 / 21.0);
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 19.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 17.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 15.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 13.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 11.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 9.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 7.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 5.0));
    q = _mm256_fmadd_pd(p, q, _mm256_set1_pd(1.0 / 3.0));
    q = _mm256_fmadd_pd(p, q, one);
    const __m256d two = _mm256_set1_pd(2.0);
    return _mm256_mul_pd(two, _mm256_mul_pd(u, q));
}

inline __m256d log8_half_pd(__m256d rd, __m128i e32_half) {
    const __m256d ln2 = _mm256_set1_pd(LN2);
    const __m256d ed = _mm256_cvtepi32_pd(e32_half);
    return _mm256_fmadd_pd(ed, ln2, log_mantissa_pd(rd));
}

} // namespace

__m256 logf8_avx2(__m256 x) {
    const __m256i ix = _mm256_castps_si256(x);

    const __m256i biased = _mm256_srli_epi32(_mm256_and_si256(ix, _mm256_set1_epi32(0x7FFFFFFF)), 23);
    __m256i e = _mm256_sub_epi32(biased, _mm256_set1_epi32(127));
    const __m256i mx = _mm256_or_si256(_mm256_set1_epi32(0x00800000),
                                       _mm256_and_si256(ix, _mm256_set1_epi32(0x007FFFFF)));
    const __m256i rbits = _mm256_or_si256(_mm256_set1_epi32(0x3F800000),
                                          _mm256_and_si256(mx, _mm256_set1_epi32(0x007FFFFF)));

    __m256 rf = _mm256_castsi256_ps(rbits);
    const __m256 sqrt2 = _mm256_set1_ps(1.4142135623730950488016887242096980785696718753769f);
    const __m256 cmp_mask_ps = _mm256_cmp_ps(rf, sqrt2, _CMP_GT_OQ);
    const __m256 rf_half = _mm256_mul_ps(rf, _mm256_set1_ps(0.5f));
    rf = _mm256_blendv_ps(rf, rf_half, cmp_mask_ps);
    e = _mm256_sub_epi32(e, _mm256_castps_si256(cmp_mask_ps));

    const __m128 r_lo = _mm256_castps256_ps128(rf);
    const __m128 r_hi = _mm256_extractf128_ps(rf, 1);
    const __m256d rd_lo = _mm256_cvtps_pd(r_lo);
    const __m256d rd_hi = _mm256_cvtps_pd(r_hi);

    const __m128i e_lo = _mm256_castsi256_si128(e);
    const __m128i e_hi = _mm256_extracti128_si256(e, 1);

    const __m256d y_lo = log8_half_pd(rd_lo, e_lo);
    const __m256d y_hi = log8_half_pd(rd_hi, e_hi);

    const __m128 y_ps_lo = _mm256_cvtpd_ps(y_lo);
    const __m128 y_ps_hi = _mm256_cvtpd_ps(y_hi);
    __m256 out = _mm256_castps128_ps256(y_ps_lo);
    out = _mm256_insertf128_ps(out, y_ps_hi, 1);
    return out;
}
