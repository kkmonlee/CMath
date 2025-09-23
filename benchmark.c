#include "cmath.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__clang__)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#if defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static double wall_time_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
#else
    return (double) clock() / (double) CLOCKS_PER_SEC;
#endif
}

static void fill_random(double *a, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        // TODO: find a better rand
        const double r = (double) rand() / (double) RAND_MAX;
        a[i] = (r * 2.0 - 1.0) * 100.0; // [-100, 100]
    }
}

static void fill_random_pos(double *a, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        const double r = (double) rand() / (double) RAND_MAX; // [0,1)
        // FIXME: [0,100] to avoid domain/zero issues in legacy exprs
        a[i] = r * 100.0;
    }
}

// y = 0.5 * (fma(u, v, w) + sqrt(w*w))
static NOINLINE double native_fused(double u, double v, double w) {
    return 0.5 * (fma(u, v, w) + sqrt(w * w));
}

static NOINLINE double native_plain(double u, double v, double w) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const volatile double uv = u * v;
    const double sum = uv + w;
    const double s = sqrt(w * w);
    return 0.5 * (sum + s);
}

static NOINLINE double expr_as_native(double a) {
    // sqrt(a^1.5 + a^2.5) = sqrt(a*sqrt(a) + (a*a)*sqrt(a)) ; a>=0 assumed
    const double sa = sqrt(a);
    const double t = a * sa + (a * a) * sa;
    return sqrt(t);
}

static NOINLINE double expr_a5_native(double a) { return a + 5.0; }
static NOINLINE double expr_a10_native(double a) { return a + (5.0 * 2.0); }
static NOINLINE double expr_a52_native(double a) { return (a + 5.0) * 2.0; }
static NOINLINE double expr_al_native(double a) { return 1.0 / (a + 1.0) + 2.0 / (a + 2.0) + 3.0 / (a + 3.0); }

static cm_program *build_program(void) {
    cm_program *p = cm_prog_create(/*num_vars=*/3, /*initial_slots=*/8);
    const uint32_t u = cm_emit_var(p, 0);
    const uint32_t v = cm_emit_var(p, 1);
    const uint32_t w = cm_emit_var(p, 2);

    const uint32_t t = cm_emit_fma(p, u, v, w); // u*v + w (fused)
    const uint32_t w2 = cm_emit_mul(p, w, w); // w*w
    const uint32_t s = cm_emit_sqrt(p, w2); // sqrt(w*w)
    const uint32_t r = cm_emit_add(p, t, s); // fma(u,v,w) + sqrt(w*w)
    const uint32_t y = cm_emit_mul_k(p, r, 0.5); // * 0.5
    cm_set_result(p, y);
    return p;
}

// sqrt(a^1.5 + a^2.5) = sqrt( sqrt(a) * a + sqrt(a) * a * a )
static cm_program *build_prog_as(void) {
    cm_program *p = cm_prog_create(/*num_vars=*/1, /*initial_slots=*/16);
    const uint32_t a = cm_emit_var(p, 0);
    const uint32_t sa = cm_emit_sqrt(p, a); // sqrt(a)
    const uint32_t a_sa = cm_emit_mul(p, a, sa); // a*sqrt(a)
    const uint32_t a2 = cm_emit_mul(p, a, a); // a*a
    const uint32_t a2_sa = cm_emit_mul(p, a2, sa); // (a*a)*sqrt(a)
    const uint32_t sum = cm_emit_add(p, a_sa, a2_sa); // a^1.5 + a^2.5
    const uint32_t y = cm_emit_sqrt(p, sum); // sqrt(...)
    cm_set_result(p, y);
    return p;
}

static cm_program *build_prog_a5(void) {
    cm_program *p = cm_prog_create(1, 8);
    const uint32_t a = cm_emit_var(p, 0);
    const uint32_t y = cm_emit_add_k(p, a, 5.0);
    cm_set_result(p, y);
    return p;
}

static cm_program *build_prog_a10(void) {
    // TODO: interpreter keeps exact ops you emit
    cm_program *p = cm_prog_create(1, 8);
    const uint32_t a = cm_emit_var(p, 0);
    const uint32_t y = cm_emit_add_k(p, a, 10.0); // acceptable for timing; structure doesn’t/shouldn't matter here
    cm_set_result(p, y);
    return p;
}

static cm_program *build_prog_a52(void) {
    cm_program *p = cm_prog_create(1, 8);
    const uint32_t a = cm_emit_var(p, 0);
    const uint32_t t = cm_emit_add_k(p, a, 5.0); // (a+5)
    const uint32_t y = cm_emit_mul_k(p, t, 2.0); // *2
    cm_set_result(p, y);
    return p;
}

static cm_program *build_prog_al(void) {
    // (1/(a+1) + 2/(a+2) + 3/(a+3)), with a in [0,100] so no singularities
    cm_program *p = cm_prog_create(1, 16);
    const uint32_t a = cm_emit_var(p, 0);

    const uint32_t d1 = cm_emit_add_k(p, a, 1.0);
    const uint32_t t1 = cm_emit_recip(p, d1);

    const uint32_t d2 = cm_emit_add_k(p, a, 2.0);
    const uint32_t t2 = cm_emit_recip(p, d2);
    const uint32_t t2s = cm_emit_mul_k(p, t2, 2.0);

    const uint32_t d3 = cm_emit_add_k(p, a, 3.0);
    const uint32_t t3 = cm_emit_recip(p, d3);
    const uint32_t t3s = cm_emit_mul_k(p, t3, 3.0);

    const uint32_t s12 = cm_emit_add(p, t1, t2s);
    const uint32_t y = cm_emit_add(p, s12, t3s);
    cm_set_result(p, y);
    return p;
}

typedef cm_program * (*prog_builder_fn)(void);

typedef double (*native1_fn)(double);

static void bench_expr(const char *label,
                       prog_builder_fn builder,
                       native1_fn native,
                       const double *A,
                       size_t N) {
    printf("\nExpression: %s\n", label);

    // native
    double t0 = wall_time_sec();
    double acc_native = 0.0;
    for (size_t i = 0; i < N; ++i) acc_native += native(A[i]);
    double t1 = wall_time_sec();
    const double dt_native = t1 - t0;

    // interpreter (no JIT)
    cm_program *p = builder();
    t0 = wall_time_sec();
    double acc_interp = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double v[1] = {A[i]};
        acc_interp += cm_eval(p, v);
    }
    t1 = wall_time_sec();
    const double dt_interp = t1 - t0;
    cm_prog_free(p);

    const double ms_native = dt_native * 1e3;
    const double ms_interp = dt_interp * 1e3;
    const double mfps_native = (double) N / 1e6 / dt_native;
    const double mfps_interp = (double) N / 1e6 / dt_interp;
    const double pct_longer = (dt_interp / dt_native - 1.0) * 100.0;

    printf("native  % .4e\t %4.0fms\t %4.0fmfps\n", acc_native, ms_native, mfps_native);
    printf("interp  % .4e\t %4.0fms\t %4.0fmfps\n", acc_interp, ms_interp, mfps_interp);
    printf("%0.2f%% longer\n", pct_longer);
}

typedef struct {
    double seconds;
    double mevals_per_s;
    double checksum;
} bench_result;

static bench_result bench_jit_scalar(const double *X, const double *Y, const double *Z, size_t N) {
    cm_program *p = build_program();
    cm_compile(p); // attempt JIT scalar

    const double t0 = wall_time_sec();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double vars[3] = {X[i], Y[i], Z[i]};
        acc += cm_eval(p, vars);
    }
    const double t1 = wall_time_sec();
    bench_result br;
    br.seconds = t1 - t0;
    br.mevals_per_s = (double) N / 1e6 / br.seconds;
    br.checksum = acc;

    printf("  JIT globally supported: %s\n", cm_jit_globally_supported() ? "yes" : "no");
    printf("  Program is jitted:     %s\n", cm_prog_is_jitted(p) ? "yes" : "no");
    cm_prog_free(p);
    return br;
}

static bench_result bench_interpreter(const double *X, const double *Y, const double *Z, size_t N) {
    cm_program *p = build_program();
    const double t0 = wall_time_sec();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double vars[3] = {X[i], Y[i], Z[i]};
        acc += cm_eval(p, vars);
    }
    const double t1 = wall_time_sec();
    bench_result br;
    br.seconds = t1 - t0;
    br.mevals_per_s = (double) N / 1e6 / br.seconds;
    br.checksum = acc;
    cm_prog_free(p);
    return br;
}

// native fused batch (SoA)
static void native_batch_fused(const double *const*inputs, size_t n, double *out) {
    const double * RESTRICT X = inputs[0];
    const double * RESTRICT Y = inputs[1];
    const double * RESTRICT Z = inputs[2];
    double * RESTRICT O = out;

    // tell compiler pointers are aligned
    // TODO: tweak 32->64 when I 64B-align the buffers
#if defined(__clang__)
    X = (const double *) __builtin_assume_aligned(X, 32);
    Y = (const double *) __builtin_assume_aligned(Y, 32);
    Z = (const double *) __builtin_assume_aligned(Z, 32);
    O = (double *) __builtin_assume_aligned(O, 32);
#pragma clang loop vectorize(enable)
#pragma clang loop interleave_count(2)
#pragma clang loop unroll_count(4)
#endif
    for (size_t i = 0; i < n; ++i) {
        double u = X[i], v = Y[i], w = Z[i];
        // inline the arithmetic so the loop is vectorizable
        // TODO: this part slows things down a LOT
        const double r = 0.5 * (__builtin_fma(u, v, w) + __builtin_sqrt(w * w));
        O[i] = r;
    }
}

static bench_result bench_native_fused(const double *X, const double *Y, const double *Z, size_t N) {
    const double *inputs[3] = {X, Y, Z};
    double *out = malloc(N * sizeof(double));
    const double t0 = wall_time_sec();
    native_batch_fused(inputs, N, out);
    const double t1 = wall_time_sec();

    // checksum
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) acc += out[i];
    free(out);

    bench_result br;
    br.seconds = t1 - t0;
    br.mevals_per_s = (double) N / 1e6 / br.seconds;
    br.checksum = acc;
    return br;
}

static bench_result bench_native_plain(const double *X, const double *Y, const double *Z, size_t N) {
    const double t0 = wall_time_sec();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        acc += native_plain(X[i], Y[i], Z[i]);
    }
    const double t1 = wall_time_sec();
    bench_result br;
    br.seconds = t1 - t0;
    br.mevals_per_s = (double) N / 1e6 / br.seconds;
    br.checksum = acc;
    return br;
}

static bench_result bench_jit_batch(const double *X, const double *Y, const double *Z, size_t N, double *out_save) {
    cm_program *p = build_program();
    cm_compile_batch(p); // attempt batch JIT

    const double *inputs[3] = {X, Y, Z};
    double *out = out_save ? out_save : (double *) malloc(N * sizeof(double));

    const double t0 = wall_time_sec();
    cm_eval_batch(p, inputs, N, out);
    const double t1 = wall_time_sec();

    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) acc += out[i];

    bench_result br;
    br.seconds = t1 - t0;
    br.mevals_per_s = (double) N / 1e6 / br.seconds;
    br.checksum = acc;

    printf("  JIT batch supported:    %s\n", cm_jit_globally_supported() ? "yes" : "no");
    printf("  Program batch jitted:   %s\n", cm_prog_is_jitted_batch(p) ? "yes" : "no");

    if (!out_save) free(out);
    cm_prog_free(p);
    return br;
}

int main(const int argc, char **argv) {
    const size_t N = (argc >= 2) ? (size_t) strtoull(argv[1], NULL, 10) : (1u << 20); // default 1,048,576
    const size_t S = (argc >= 3) ? (size_t) strtoull(argv[2], NULL, 10) : 10000; // samples for accuracy

    double *X = malloc(N * sizeof(double));
    double *Y = malloc(N * sizeof(double));
    double *Z = malloc(N * sizeof(double));
    if (!X || !Y || !Z) {
        fprintf(stderr, "Allocation failed\n");
        free(X);
        free(Y);
        free(Z);
        return 1;
    }
    fill_random(X, N, 1234);
    fill_random(Y, N, 5678);
    fill_random(Z, N, 9012);

    // accuracy vs native fused C (scalar)
    const size_t samples = (S < N) ? S : N;
    double max_abs_err = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < samples; ++i) {
        cm_program *p = build_program();
        cm_compile(p);
        const double vars[3] = {X[i], Y[i], Z[i]};
        const double got = cm_eval(p, vars);
        const double ref = native_fused(X[i], Y[i], Z[i]);
        const double err = fabs(got - ref);
        if (err > max_abs_err) max_abs_err = err;
        sum_err += err;
        cm_prog_free(p);
    }

    printf("Accuracy vs native fused C (over %zu samples):\n", samples);
    printf("  max abs error: %.3e, mean abs error: %.3e\n", max_abs_err, sum_err / (double) samples);

    // throughput (scalar)
    printf("\nThroughput (scalar) over %zu evaluations:\n", N);
    const bench_result r_jit = bench_jit_scalar(X, Y, Z, N);
    const bench_result r_interp = bench_interpreter(X, Y, Z, N);
    const bench_result r_nfuse = bench_native_fused(X, Y, Z, N);
    const bench_result r_nplain = bench_native_plain(X, Y, Z, N);

    printf("\nScalar results:\n");
    printf("  JIT (fused):            %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_jit.mevals_per_s, r_jit.seconds, r_jit.checksum);
    printf("  Interpreter:            %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_interp.mevals_per_s, r_interp.seconds, r_interp.checksum);
    printf("  Native C (fused fma):   %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_nfuse.mevals_per_s, r_nfuse.seconds, r_nfuse.checksum);
    printf("  Native C (plain mul+add, contraction off): %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_nplain.mevals_per_s, r_nplain.seconds, r_nplain.checksum);

    // throughput (batch)
    printf("\nThroughput (batch/SoA) over %zu evaluations:\n", N);
    double *out_jit = malloc(N * sizeof(double));
    const bench_result r_jitb = bench_jit_batch(X, Y, Z, N, out_jit);
    const double *inputs[3] = {X, Y, Z};
    double *out_native = malloc(N * sizeof(double));
    const double t0b = wall_time_sec();
    native_batch_fused(inputs, N, out_native);
    const double t1b = wall_time_sec();
    bench_result r_nfb;
    r_nfb.seconds = t1b - t0b;
    r_nfb.mevals_per_s = (double) N / 1e6 / r_nfb.seconds;
    double accb = 0.0;
    for (size_t i = 0; i < N; ++i) accb += out_native[i];
    r_nfb.checksum = accb;

    // accu
    double max_abs_err_b = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double err = fabs(out_jit[i] - out_native[i]);
        if (err > max_abs_err_b) max_abs_err_b = err;
    }

    printf("\nBatch results (SoA):\n");
    printf("  JIT batch (fused):      %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_jitb.mevals_per_s, r_jitb.seconds, r_jitb.checksum);
    printf("  Native batch (fused):   %8.2f M eval/s, time %.3f s, checksum %.6f\n",
           r_nfb.mevals_per_s, r_nfb.seconds, r_nfb.checksum);
    printf("  Batch max abs error vs native: %.3e\n", max_abs_err_b);

    printf("\nSpeedups:\n");
    printf("  Scalar JIT vs native fused:    %.2fx\n", r_jit.mevals_per_s / r_nfuse.mevals_per_s);
    printf("  Scalar JIT vs interpreter:     %.2fx\n", r_jit.mevals_per_s / r_interp.mevals_per_s);
    printf("  Batch JIT vs native batch:     %.2fx\n", r_jitb.mevals_per_s / r_nfb.mevals_per_s);
    printf("  Batch JIT vs scalar JIT:       %.2fx\n", r_jitb.mevals_per_s / r_jit.mevals_per_s);

    printf("Key metric: Batch JIT/native = %.2fx (%s)\n",
           r_jitb.mevals_per_s / r_nfb.mevals_per_s,
           (r_jitb.mevals_per_s >= r_nfb.mevals_per_s) ? "JIT faster" : "native faster");

    printf("\nLegacy expression microbenchmarks (single var a, N=%zu):\n", N);
    double *A = malloc(N * sizeof(double));
    if (!A) {
        fprintf(stderr, "Allocation failed for A\n");
        free(out_jit);
        free(out_native);
        free(X);
        free(Y);
        free(Z);
        return 1;
    }
    // FIXME: restrict a >= 0 to avoid domain issues
    fill_random_pos(A, N, 4242);

    bench_expr("sqrt(a^1.5+a^2.5)", build_prog_as, expr_as_native, A, N);
    bench_expr("a+5", build_prog_a5, expr_a5_native, A, N);
    bench_expr("a+(5*2)", build_prog_a10, expr_a10_native, A, N);
    bench_expr("(a+5)*2", build_prog_a52, expr_a52_native, A, N);
    bench_expr("(1/(a+1)+2/(a+2)+3/(a+3))",
               build_prog_al, expr_al_native, A, N);

    free(A);
    free(out_jit);
    free(out_native);
    free(X);
    free(Y);
    free(Z);
    return 0;
}
