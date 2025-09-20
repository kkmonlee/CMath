#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../cmath.h"
#include "../cm_vector.h"
#include "../cmath.c"

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
    cm_expr* expr;

    expr = cm_compile("1+2", NULL, 0, &error);
    TEST_ASSERT(expr != NULL && error == 0, "Should parse simple expression.");
    cm_free(expr);

    expr = cm_compile("sin(pi/2)", NULL, 0, &error);
    TEST_ASSERT(expr != NULL && error == 0, "Should parse functions and constants.");
    cm_free(expr);
    
    // Test for syntax error
    expr = cm_compile("1+*2", NULL, 0, &error);
    TEST_ASSERT(expr == NULL && error != 0, "Should fail on syntax error.");
    cm_free(expr);

    // Test for undefined variable
    expr = cm_compile("x+1", NULL, 0, &error);
    TEST_ASSERT(expr == NULL && error != 0, "Should fail on undefined variable.");
    cm_free(expr);
}

void suite_evaluator_correctness() {
    double x = 3.14;
    double y = 2.71;
    cm_variable vars[] = {
        {"x", &x, CM_VAR},
        {"y", &y, CM_VAR}
    };
    const char* expr_str = "sin(x) * (y+5)/log(x^2)";

    int error = 0;
    cm_expr *n = cm_compile(expr_str, vars, 2, &error);
    if (!n) {
        TEST_ASSERT(0, "Failed to compile the test expression.");
        return;
    }

    double val_eval = cm_eval(n, &error);
    TEST_ASSERT(error == 0, "cm_eval should execute without error.");
    
    double val_fast = cm_eval_fast(n);
    TEST_ASSERT(is_close(val_eval, val_fast), "cm_eval result must match cm_eval_fast.");

    #ifdef __GNUC__
    double val_goto = cm_eval_computed_goto(n);
    TEST_ASSERT(is_close(val_eval, val_goto), "cm_eval result must match cm_eval_computed_goto.");
    #endif

    cm_bytecode* bc = cm_compile_bytecode(n, 2);
    if (bc) {
        double vars_arr[] = {x, y};
        double val_bc = cm_eval_bytecode(bc, vars_arr);
        TEST_ASSERT(is_close(val_eval, val_bc), "cm_eval result must match bytecode VM.");
        cm_bytecode_free(bc);
    } else {
        TEST_ASSERT(0, "Bytecode compilation failed.");
    }
    
    #if (defined(__x86_64__) || defined(_M_X64)) && !defined(__MINGW32__)
    if (n->jit_code) {
        cm_jit_code* jit = n->jit_code;
        if (jit->compiled_func) {
             double vars_arr[] = {x, y};
             double val_jit = jit->compiled_func(vars_arr);
             TEST_ASSERT(is_close(val_eval, val_jit), "cm_eval result must match JIT compiled code.");
        }
    }
    #endif

    cm_free(n);
}

void suite_optimizer_correctness() {
    double x = 5.0;
    cm_variable vars[] = {{"x", &x, CM_VAR}};
    int error = 0;

    // Test: x * 1 = x
    cm_expr* expr_opt = cm_compile("x * 1.0", vars, 1, &error);
    cm_expr* expr_base = cm_compile("x", vars, 1, &error);
    TEST_ASSERT(is_close(cm_eval(expr_opt, &error), cm_eval(expr_base, &error)), "Optimizer: x * 1.0 == x");
    cm_free(expr_opt);
    cm_free(expr_base);

    // Test: x + 0 = x
    expr_opt = cm_compile("x + 0.0", vars, 1, &error);
    expr_base = cm_compile("x", vars, 1, &error);
    TEST_ASSERT(is_close(cm_eval(expr_opt, &error), cm_eval(expr_base, &error)), "Optimizer: x + 0.0 == x");
    cm_free(expr_opt);
    cm_free(expr_base);

    // Test: 2 + 3 = 5
    expr_opt = cm_compile("2 + 3", NULL, 0, &error);
    TEST_ASSERT(is_close(cm_eval(expr_opt, &error), 5.0), "Optimizer: constant folding 2+3==5.");
    cm_free(expr_opt);
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
    #if defined(CM_HAVE_AVX2) || defined(CM_HAVE_NEON)
    const int N = 16;
    double in[N], out[N], expected[N];

    // --- Test exp ---
    for (int i = 0; i < N; ++i) in[i] = (double)i / 4.0 - 2.0;

    #if defined(CM_HAVE_AVX2)
    vec_exp_avx2(in, out, N);
    #elif defined(CM_HAVE_NEON)
    vec_exp_neon(in, out, N);
    #endif
    
    for (int i = 0; i < N; ++i) expected[i] = exp(in[i]);
    
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(is_close(out[i], expected[i]), "Vector EXP kernel mismatch at index %d", i);
    }

    // --- Test sin ---
    #if defined(CM_HAVE_AVX2)
    for (int i = 0; i < N; ++i) in[i] = (double)i * M_PI / 8.0;
    vec_sin_avx2(in, out, N);
    for (int i = 0; i < N; ++i) expected[i] = sin(in[i]);
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(is_close(out[i], expected[i]), "Vector SIN kernel mismatch at index %d", i);
    }
    #endif
    
    #endif // Vectorization enabled
}

int main() {
    run_test_suite(suite_parser, "Parser");
    run_test_suite(suite_evaluator_correctness, "Evaluator Correctness");
    run_test_suite(suite_optimizer_correctness, "Optimizer Correctness");
    run_test_suite(suite_vector_helpers, "Vector Helpers");
    run_test_suite(suite_vector_kernels, "Vector Kernels");

    printf("\n--- Test Summary ---\n");
    printf("Passed: %d, Failed: %d\n", tests_passed, tests_failed);
    
    if (tests_failed > 0) {
        return 1;
    }
    return 0;
}