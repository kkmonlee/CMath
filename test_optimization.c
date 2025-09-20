#include "cmath.h"
#include <stdio.h>

int main() {
    int error = 0;

    // Test constant folding - this should be optimized to a single constant
    cm_expr *expr1 = cm_compile("5*2+3", NULL, 0, &error);
    if (expr1) {
        printf("5*2+3 = %f\n", cm_eval(expr1, NULL));
        printf("Expression type after optimization: %s\n",
               (expr1->type & 0x18) == 0x08 ? "CONST" : "OTHER");
        cm_free(expr1);
    }

    // Test with variable - should not be constant
    double x = 5.0;
    cm_variable vars[] = {{"x", &x}};
    cm_expr *expr2 = cm_compile("x*2+3", vars, 1, &error);
    if (expr2) {
        printf("x*2+3 (x=5) = %f\n", cm_eval(expr2, NULL));
        printf("Expression type: %s\n",
               (expr2->type & 0x18) == 0x08 ? "CONST" : "OTHER");
        cm_free(expr2);
    }

    return 0;
}