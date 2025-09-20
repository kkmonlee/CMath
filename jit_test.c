#include "cmath.h"
#include <stdio.h>
#include <time.h>

#define ITERATIONS 100000000

int main() {
    double x = 5.0;
    cm_variable vars[] = {{"x", &x}};
    int error = 0;

    // Test simple addition: x + 5
    cm_expr *expr = cm_compile("x+5", vars, 1, &error);
    if (!expr) {
        printf("Compilation failed\n");
        return 1;
    }

    printf("Testing JIT vs interpreter performance...\n");
    printf("Expression: x+5 where x=5\n");

    // Warm up
    for (int i = 0; i < 1000; i++) {
        cm_eval(expr, NULL);
    }

    clock_t start = clock();
    double result = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        result += cm_eval(expr, NULL);
    }
    clock_t end = clock();

    printf("Result: %f (should be 10.0 * %d = %f)\n", result, ITERATIONS, 10.0 * ITERATIONS);
    printf("Time: %dms\n", (int)((end - start) * 1000 / CLOCKS_PER_SEC));
    printf("Rate: %.1f million evals/sec\n", ITERATIONS / 1000.0 / ((end - start) * 1000.0 / CLOCKS_PER_SEC));

    cm_free(expr);
    return 0;
}