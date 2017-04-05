//
// Created by aa on 19/12/16.
//

#ifndef __CMATH_H__
#define __CMATH_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*cm_fun0)(void);

typedef double (*cm_fun1)(double);

typedef double (*cm_fun2)(double, double);

typedef struct cm_expr {
    struct cm_expr *left, *right;
    int type;
    union {
        double value;
        const double *bound;
        cm_fun0 f0;
        cm_fun1 f1;
        cm_fun2 f2;
    };
} cm_expr;

enum {
    CM_FUNCTION0 = -1, CM_VARIABLE = 0, CM_FUNCTION1 = 1, CM_FUNCTION2 = 2
};

typedef struct cm_variable {
    const char *name;
    const void *value;
    int type;
} cm_variable;

// Parses the input expression, evaluates it and frees it
double cm_interp(const char *expression, int *error);

// Parses the input expression and binds variables
cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error);

// Evaluate the expression
double cm_eval(const cm_expr *n);

// Prints debugging information
void cm_print(const cm_expr *n);

// Frees expression
void cm_free(cm_expr *n);

#ifdef __cplusplus
}
#endif
#endif //__CMATH_H__
