//
// Created by aa on 02/04/17.
//

#include <stdio.h>
#include <time.h>
#include <math.h>
#include "cmath.h"

#define loops 10000

void bench(const char *expr, cm_fun1 func) {
    int i, j;
    volatile double d;
    double tmp;
    clock_t start;

    cm_variable lk = {"a", &tmp};

    printf("Expression: %s\n", expr);

    printf("native ");
    start = clock();
    d = 0;
    for (j = 0; j < loops; ++j) {
        for (i = 0; i < loops; ++i) {
            tmp = i;
            d += func(tmp);
        }
    }
    const int nelapsed = (clock() - start) * 1000 / CLOCKS_PER_SEC;

    printf(" %.5g", d);
    if (nelapsed)
        printf("\t%5dms\t%5dmfps\n", nelapsed, loops * loops / nelapsed / 1000);
    else
        printf("\tinf\n");

    printf("interp ");
    int error = 0;
    cm_expr *n = cm_compile(expr, &lk, 1, &error);
    if (!n) {
        printf("Compilation failed with error %d\n", error);
        return;
    }
    start = clock();
    d = 0;
    // Loop unrolling for better performance
    for (j = 0; j < loops; ++j) {
        for (i = 0; i < loops; i += 8) {
            // Unroll 8 iterations at once
            tmp = i; d += cm_eval(n, NULL);
            tmp = i+1; d += cm_eval(n, NULL);
            tmp = i+2; d += cm_eval(n, NULL);
            tmp = i+3; d += cm_eval(n, NULL);
            tmp = i+4; d += cm_eval(n, NULL);
            tmp = i+5; d += cm_eval(n, NULL);
            tmp = i+6; d += cm_eval(n, NULL);
            tmp = i+7; d += cm_eval(n, NULL);
        }
    }

    const int eelapsed = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    cm_free(n);

    printf(" %.5g", d);
    if (eelapsed)
        printf("\t%5dms\t%5dmfps\n", eelapsed, loops * loops / eelapsed / 1000);
    else
        printf("\tinf\n");

    printf("%.2f%% longer\n", (((double) eelapsed / nelapsed) - 1.0) * 100.0);
    printf("\n");
}

double a5(double a) {
    return a + 5;
}

double a52(double a) {
    return (a + 5) * 2;
}

double a10(double a) {
    return a + (5 * 2);
}

double as(double a) {
    return sqrt(pow(a, 1.5) + pow(a, 2.5));
}

double al(double a) {
    return (1 / (a + 1) + 2 / (a + 2) + 3 / (a + 3));
}

int main(int argc, char *argv[]) {

    bench("sqrt(a^1.5+a^2.5)", as);
    bench("a+5", a5);
    bench("a+(5*2)", a10);
    bench("(a+5)*2", a52);
    bench("(1/(a+1)+2/(a+2)+3/(a+3))", al);

    return 0;
}