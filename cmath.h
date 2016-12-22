//
// Created by aa on 19/12/16.
//

#ifndef __CMATH_H__
#define __CMATH_H__

typedef double (*cm_fun1) (double);
typedef double (*cm_fun2) (double, double);

typedef struct cm_expr {
    struct cm_expr *left, *right;
    union {double value; cm_fun1 f1; cm_fun2 f2;};
    const double *bound;
} cm_expr;

typedef struct {
    const char *name;
    const double value;
} cm_variable;

// Parses the input expression, evaluates it and frees it
double cm_interp(const char *expression, int *error);

// Parses the input expression and binds variables
cm_expr *cm_compile(const char *expression, const cm_variable *lookup, int lookup_len, int *error);

// Evaluate the expression
double cm_eval(const cm_expr *n);

// Prints debugging information
void cm_print(const cm_expr *n);

// Frees expression
void cm_free(cm_expr *n);

#endif //__CMATH_H__
