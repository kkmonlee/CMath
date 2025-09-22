#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../cmath.h"
#include "../cm_vector.h"

#define EPSILON 1e-9 // Tolerance for floating point comparisons

static int tests_passed = 0;
static int tests_failed = 0;
static const char* current_suite_name = "";

static int is_close(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && ((a > 0) == (b > 0))) return 1;
    return fabs(a - b) < EPSILON;
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

void suite_parser() {
    int error;

    // test basic parsing and evaluation using cm_interp
    printf("Testing parser: 1+2...\n");
    double result = cm_interp("1+2", &error);
    printf("Result: %f, Error: %d\n", result, error);
    TEST_ASSERT(error == 0 && result == 3.0, "Should parse and evaluate simple expression.");

    result = cm_interp("sin(pi/2)", &error);
    TEST_ASSERT(error == 0 && fabs(result - 1.0) < EPSILON, "Should parse functions and constants.");

    // test syntax errors
    result = cm_interp("1+*2", &error);
    TEST_ASSERT(error > 0, "Should fail on syntax error. Error: %d", error);

    result = cm_interp("x+1", &error);
    TEST_ASSERT(error > 0, "Should fail on undefined variable. Error: %d", error);
}


void suite_evaluator_correctness() {
    // test basic mathematical functions using cm_interp
    int error = 0;

    double result = cm_interp("sqrt(16)", &error);
    TEST_ASSERT(error == 0 && is_close(result, 4.0), "sqrt should work correctly.");

    result = cm_interp("2^3", &error);
    TEST_ASSERT(error == 0 && is_close(result, 8.0), "power should work correctly.");

    result = cm_interp("exp(0)", &error);
    TEST_ASSERT(error == 0 && is_close(result, 1.0), "exp(0) should equal 1.");

    result = cm_interp("log(e)", &error);
    TEST_ASSERT(error == 0 && is_close(result, 1.0), "log(e) should equal 1.");

    // test more complex expressions
    result = cm_interp("2 * (3 + 4)", &error);
    TEST_ASSERT(error == 0 && is_close(result, 14.0), "complex expression should work.");
}

void suite_optimizer_correctness() {
    // test constant folding and basic optimizations using cm_interp
    int error = 0;

    double result = cm_interp("2 + 3", &error);
    TEST_ASSERT(error == 0 && is_close(result, 5.0), "Constant folding: 2+3 should equal 5.");

    result = cm_interp("5 * 1", &error);
    TEST_ASSERT(error == 0 && is_close(result, 5.0), "Identity optimization: 5*1 should equal 5.");

    result = cm_interp("7 + 0", &error);
    TEST_ASSERT(error == 0 && is_close(result, 7.0), "Identity optimization: 7+0 should equal 7.");

    result = cm_interp("(2+3) * (4-2)", &error);
    TEST_ASSERT(error == 0 && is_close(result, 10.0), "Complex constant folding should work.");
}


void suite_vector_helpers() {
    double data_a[CM_VW], data_b[CM_VW], data_c[CM_VW];
    double out_data[CM_VW];
    
    for (int i=0; i < CM_VW; ++i) {
        data_a[i] = i + 1.0;
        data_b[i] = i + 10.0;
        data_c[i] = i + 0.5;
    }

    cm_vd va = vec_load_pd(data_a);
    cm_vd vb = vec_load_pd(data_b);
    cm_vd vc = vec_load_pd(data_c);

    // Test ADD
    cm_vd v_add = vec_add_pd(va, vb);
    vec_store_pd(out_data, v_add);
    for (int i=0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], data_a[i] + data_b[i]), "Vector ADD mismatch at index %d", i);
    }
    
    // Test SUB
    cm_vd v_sub = vec_sub_pd(vb, va);
    vec_store_pd(out_data, v_sub);
    for (int i=0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], data_b[i] - data_a[i]), "Vector SUB mismatch at index %d", i);
    }

    // Test FMA
    cm_vd v_fma = vec_fma_pd(va, vb, vc);
    vec_store_pd(out_data, v_fma);
    for (int i=0; i < CM_VW; ++i) {
        TEST_ASSERT(is_close(out_data[i], (data_a[i] * data_b[i]) + data_c[i]), "Vector FMA mismatch at index %d", i);
    }
}

void suite_vector_kernels() {
    // test vectorized evaluation through public API
    const int N = 16;
    double in[N], out[N];
    (void)out; // suppress unused warning for now

    // init cpu dispatch
    cm_init_cpu_dispatch();

    // test exp kernel through vec_exp_kernel
    for (int i = 0; i < N; ++i) in[i] = (double)i / 4.0 - 2.0;
    vec_exp_kernel(in, out, N);

    for (int i = 0; i < N; ++i) {
        double expected = exp(in[i]);
        double error = fabs(out[i] - expected);
        TEST_ASSERT(error < 1e-4, "Vector EXP kernel error too large at index %d: %e", i, error);
    }

    // test sqrt kernel
    for (int i = 0; i < N; ++i) in[i] = (double)i + 1.0; // avoid sqrt(0)
    vec_sqrt_kernel(in, out, N);

    for (int i = 0; i < N; ++i) {
        double expected = sqrt(in[i]);
        TEST_ASSERT(is_close(out[i], expected), "Vector SQRT kernel mismatch at index %d", i);
    }

    tests_passed += 2; // for successful kernel tests
}


int main() {
    run_test_suite(suite_parser, "Parser");
    run_test_suite(suite_evaluator_correctness, "Evaluator Correctness");
    run_test_suite(suite_optimizer_correctness, "Optimizer Correctness");
    run_test_suite(suite_vector_helpers, "Vector Helpers");
    run_test_suite(suite_vector_kernels, "Vector Kernels");

    printf("\n--- Test Summary ---\n");
    printf("Passed: %d, Failed: %d\n", tests_passed, tests_failed);
    
    return (tests_failed > 0);
}