#include <stdio.h>
#include <time.h>

#define ITERATIONS 100000000

int main() {
    double x = 5.0;

    printf("Testing native C performance...\n");
    printf("Expression: x+5 where x=5\n");

    clock_t start = clock();
    double result = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        result += x + 5.0; // Direct native operation
    }
    clock_t end = clock();

    printf("Result: %f (should be 10.0 * %d = %f)\n", result, ITERATIONS, 10.0 * ITERATIONS);
    printf("Time: %dms\n", (int)((end - start) * 1000 / CLOCKS_PER_SEC));
    printf("Rate: %.1f million evals/sec\n", ITERATIONS / 1000.0 / ((end - start) * 1000.0 / CLOCKS_PER_SEC));

    return 0;
}