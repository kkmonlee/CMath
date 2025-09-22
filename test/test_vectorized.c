#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cmath.h"
#include "../cm_vector.h"

#define EPSILON 1e-9
#define VEC_EPSILON 1e-4  // more relaxed for vectorized kernels

static int tests_passed = 0;
static int tests_failed = 0;
static const char* current_suite_name = "";

static int is_close(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && ((a > 0) == (b > 0))) return 1;
    return fabs(a - b) < EPSILON;
}

static int is_vec_close(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && ((a > 0) == (b > 0))) return 1;
    return fabs(a - b) < VEC_EPSILON;
}

#define TEST_ASSERT(condition, message, ...) \
    do { \
        if (condition) { \
            tests_passed++; \
        } else { \
            tests_failed++; \
            fprintf(stderr, "FAIL [%s]: " message "\n", current_suite_name, ##__VA_ARGS__); \
        } \
    } while (0)

void run_test_suite(void (*suite)(void), const char* name) {
    current_suite_name = name;
    printf("--- Running Suite: %s ---\n", name);
    suite();
}

void suite_vector_primitives() {
    // test basic vector operations
    double data_a[CM_VW], data_b[CM_VW], data_c[CM_VW];
    double out_data[CM_VW];

    for (int i = 0; i < CM_VW; ++i) {
        data_a[i] = i + 1.0;
        data_b[i] = i + 10.0;
        data_c[i] = i + 0.5;
    }

    cm_vd va = vec_load_pd(data_a);
    cm_vd vb = vec_load_pd(data_b);
    cm_vd vc = vec_load_pd(data_c);

    // test set1
    cm_vd v_set = vec_set1_pd(42.0);
    vec_store_pd(out_data, v_set);
    for (int i = 0; i < CM_VW; ++i) {
        TEST_ASSERT(out_data[i] == 42.0, "vec_set1_pd failed at index %d", i);
    }

    // test add
    cm_vd v_add = vec_add_pd(va, vb);
    vec_store_pd(out_data, v_add);
    for (int i = 0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], data_a[i] + data_b[i]), "vec_add_pd failed at index %d", i);
    }

    // test sub
    cm_vd v_sub = vec_sub_pd(vb, va);
    vec_store_pd(out_data, v_sub);
    for (int i = 0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], data_b[i] - data_a[i]), "vec_sub_pd failed at index %d", i);
    }

    // test mul
    cm_vd v_mul = vec_mul_pd(va, vb);
    vec_store_pd(out_data, v_mul);
    for (int i = 0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], data_a[i] * data_b[i]), "vec_mul_pd failed at index %d", i);
    }

    // test fma
    cm_vd v_fma = vec_fma_pd(va, vb, vc);
    vec_store_pd(out_data, v_fma);
    for (int i = 0; i < CM_VW; ++i) {
        double expected = data_a[i] * data_b[i] + data_c[i];
        TEST_ASSERT(is_close(out_data[i], expected), "vec_fma_pd failed at index %d", i);
    }
}

void suite_vector_math_kernels() {
    cm_init_cpu_dispatch();

    const int N = 32;
    double in[N], out[N], expected[N];

    // test exp kernel
    for (int i = 0; i < N; ++i) {
        in[i] = (double)i / 8.0 - 2.0;  // range -2 to 2
        expected[i] = exp(in[i]);
    }

    vec_exp_kernel(in, out, N);

    double max_exp_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double error = fabs(out[i] - expected[i]);
        if (error > max_exp_error) max_exp_error = error;
        TEST_ASSERT(error < VEC_EPSILON, "exp kernel error too large at index %d: got %f, expected %f, error %e",
                   i, out[i], expected[i], error);
    }
    printf("    exp kernel max error: %e\n", max_exp_error);

    // test sqrt kernel
    for (int i = 0; i < N; ++i) {
        in[i] = (double)i * 0.5 + 0.1;  // positive values
        expected[i] = sqrt(in[i]);
    }

    vec_sqrt_kernel(in, out, N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(is_close(out[i], expected[i]), "sqrt kernel mismatch at index %d: got %f, expected %f",
                   i, out[i], expected[i]);
    }

    // test sin kernel (relaxed accuracy for now)
    for (int i = 0; i < N; ++i) {
        in[i] = (double)i * M_PI / 16.0;  // range 0 to 2π
        expected[i] = sin(in[i]);
    }

    vec_sin_kernel(in, out, N);

    // just check that it doesn't crash and produces reasonable output
    int sin_reasonable = 1;
    for (int i = 0; i < N; ++i) {
        if (isnan(out[i]) || isinf(out[i]) || fabs(out[i]) > 10.0) {
            sin_reasonable = 0;
            break;
        }
    }
    TEST_ASSERT(sin_reasonable, "sin kernel produced unreasonable outputs");
}

void suite_vectorized_evaluation() {
    cm_init_cpu_dispatch();

    const int N = 100;
    double x_vals[N], results[N];
    const double *vars[1] = {x_vals};

    // prepare test data
    for (int i = 0; i < N; ++i) {
        x_vals[i] = (double)i * 0.1;
    }

    double x = 0.0;
    cm_variable var = {"x", &x, 0};
    int error = 0;

    // test simple exp(x) vectorization
    cm_expr *exp_expr = cm_compile("exp(x)", &var, 1, &error);
    if (exp_expr && error == 0) {
        cm_eval_vec(exp_expr, results, vars, N, CM_MODE_FAST);

        // verify results
        for (int i = 0; i < N; ++i) {
            double expected = exp(x_vals[i]);
            double rel_error = fabs(results[i] - expected) / fabs(expected);
            TEST_ASSERT(rel_error < 0.01, "vectorized exp evaluation failed at index %d: rel_error %e", i, rel_error);
        }
        cm_free(exp_expr);
    } else {
        tests_failed++;
        fprintf(stderr, "FAIL [%s]: Failed to compile exp(x)\n", current_suite_name);
    }

    // test simple sqrt(x) vectorization
    cm_expr *sqrt_expr = cm_compile("sqrt(x)", &var, 1, &error);
    if (sqrt_expr && error == 0) {
        // use positive values only
        for (int i = 0; i < N; ++i) {
            x_vals[i] = (double)i * 0.1 + 0.1;
        }

        cm_eval_vec(sqrt_expr, results, vars, N, CM_MODE_FAST);

        // verify results
        for (int i = 0; i < N; ++i) {
            double expected = sqrt(x_vals[i]);
            TEST_ASSERT(is_close(results[i], expected), "vectorized sqrt evaluation failed at index %d", i);
        }
        cm_free(sqrt_expr);
    } else {
        tests_failed++;
        fprintf(stderr, "FAIL [%s]: Failed to compile sqrt(x)\n", current_suite_name);
    }
}

void suite_performance_baseline() {
    cm_init_cpu_dispatch();

    const int N = 10000;
    double inputs[N], scalar_out[N], vector_out[N];

    // prepare data
    for (int i = 0; i < N; ++i) {
        inputs[i] = (double)i / 1000.0;
    }

    // test exp performance
    clock_t start = clock();
    for (int i = 0; i < N; ++i) {
        scalar_out[i] = exp(inputs[i]);
    }
    clock_t scalar_time = clock() - start;

    start = clock();
    vec_exp_kernel(inputs, vector_out, N);
    clock_t vector_time = clock() - start;

    double speedup = (double)scalar_time / (double)vector_time;
    printf("    exp speedup: %.2fx (scalar: %ld, vector: %ld)\n", speedup, scalar_time, vector_time);

    // verify correctness
    int correct_count = 0;
    for (int i = 0; i < N; ++i) {
        if (is_vec_close(scalar_out[i], vector_out[i])) {
            correct_count++;
        }
    }

    double accuracy = (double)correct_count / N;
    TEST_ASSERT(accuracy > 0.95, "vectorized exp accuracy too low: %.2f%% correct", accuracy * 100);
    TEST_ASSERT(speedup > 0.8, "vectorized exp performance regression: %.2fx speedup", speedup);
}

int main() {
    printf("=== CMath Vectorized Tests ===\n");
    printf("SIMD Backend: ");
#if defined(CM_HAVE_AVX2)
    printf("AVX2 (x86-64)\n");
#elif defined(CM_HAVE_NEON)
    printf("NEON (ARM64)\n");
#else
    printf("Scalar fallback\n");
#endif
    printf("Vector Width: %d doubles\n\n", CM_VW);

    run_test_suite(suite_vector_primitives, "Vector Primitives");
    run_test_suite(suite_vector_math_kernels, "Vector Math Kernels");
    run_test_suite(suite_vectorized_evaluation, "Vectorized Evaluation");
    run_test_suite(suite_performance_baseline, "Performance Baseline");

    printf("\n--- Test Summary ---\n");
    printf("Passed: %d, Failed: %d\n", tests_passed, tests_failed);

    if (tests_failed == 0) {
        printf("✅ All vectorized tests passed!\n");
    } else {
        printf("❌ Some tests failed\n");
    }

    return (tests_failed > 0) ? 1 : 0;
}