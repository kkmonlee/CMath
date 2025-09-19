//
// Created by aa on 02/04/17.
//

#include "cmath.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    const char *c = "sqrt(5^2+7^2+11^2+(8-2)^2)";
    double r = cm_interp(c, 0);
    printf("Expression: \n\t%s\nevaluates to:\n\t%f\n", c, r);

    return 0;
}