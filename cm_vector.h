#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define CM_HAVE_AVX2 1
#elif defined(__aarch64__)
  #include <arm_neon.h>
  #define CM_HAVE_NEON 1
#else
  #define CM_HAVE_SCALAR 1
#endif

// vector width in doubles
#if defined(CM_HAVE_AVX2)
  #define CM_VW 4
  typedef __m256d cm_vd;
#elif defined(CM_HAVE_NEON)
  #define CM_VW 2
  typedef float64x2_t cm_vd;
#else
  #define CM_VW 1
  typedef double cm_vd;
#endif

// vector helpers
static inline cm_vd vec_set1_pd(double x) {
#if defined(CM_HAVE_AVX2)
  return _mm256_set1_pd(x);
#elif defined(CM_HAVE_NEON)
  return vdupq_n_f64(x);
#else
  return x;
#endif
}

static inline cm_vd vec_load_pd(const double *p) {
#if defined(CM_HAVE_AVX2)
  return _mm256_loadu_pd(p);
#elif defined(CM_HAVE_NEON)
  return vld1q_f64(p);
#else
  return *p;
#endif
}

static inline void vec_store_pd(double *p, cm_vd v) {
#if defined(CM_HAVE_AVX2)
  _mm256_storeu_pd(p, v);
#elif defined(CM_HAVE_NEON)
  vst1q_f64(p, v);
#else
  *p = v;
#endif
}

static inline cm_vd vec_add_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_add_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vaddq_f64(a,b);
#else
  return a + b;
#endif
}

static inline cm_vd vec_sub_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_sub_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vsubq_f64(a,b);
#else
  return a - b;
#endif
}

static inline cm_vd vec_mul_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_mul_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vmulq_f64(a,b);
#else
  return a * b;
#endif
}

static inline cm_vd vec_fma_pd(cm_vd a, cm_vd b, cm_vd c) {
#if defined(CM_HAVE_AVX2)
  return _mm256_fmadd_pd(a,b,c);
#elif defined(CM_HAVE_NEON)
  #if defined(__ARM_FEATURE_FMA)
    return vfmaq_f64(c, a, b);
  #else
    return vaddq_f64(vmulq_f64(a,b), c);
  #endif
#else
  return a*b + c;
#endif
}

// extract lane for scalar fallback
static inline double vec_get_lane(cm_vd v, int lane) {
#if defined(CM_HAVE_AVX2)
    return ((double*)&v)[lane];
#elif defined(CM_HAVE_NEON)
    return (lane == 0) ? vgetq_lane_f64(v, 0) : vgetq_lane_f64(v, 1);
#else
    return v;
#endif
}

// polynomial evaluation using horner with fma
static inline cm_vd vec_poly_horner(const double *coeffs, int deg, cm_vd x) {
    cm_vd acc = vec_set1_pd(coeffs[deg]);
    for (int i = deg-1; i >= 0; --i) {
        acc = vec_fma_pd(acc, x, vec_set1_pd(coeffs[i]));
    }
    return acc;
}