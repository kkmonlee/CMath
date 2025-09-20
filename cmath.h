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

typedef double (*cm_fun3)(double, double, double);

typedef double (*cm_fun4)(double, double, double, double);

typedef double (*cm_fun5)(double, double, double, double, double);

typedef double (*cm_fun6)(double, double, double, double, double, double);

typedef double (*cm_fun7)(double, double, double, double, double, double, double);

typedef union {
    cm_fun0 f0; cm_fun1 f1; cm_fun2 f2; cm_fun3 f3; cm_fun4 f4; cm_fun5 f5; cm_fun6 f6; cm_fun7 f7;
} cm_fun;

// Optimization structures (opaque pointers for internal use)

typedef struct cm_expr {
    struct cm_expr *left, *right;
    int type;
    union {
        double value;
        const double *bound;
        cm_fun fun;
    };
    int member_count;
    void *jit_code; // JIT compiled function pointer
    void *bytecode; // Bytecode for VM execution
    void *pattern; // Pattern specialization info
    unsigned char optimization_flags; // Bit flags for optimization status
    struct cm_expr *members[];
} cm_expr;

#define CM_MASK_ARIT 0x00000007 /* Three bits, Arity, max is 8 */
#define CM_FLAG_TYPE 0x00000018 /* Two bits, 1 = constant, 2 = variable, 3 = function */

enum {
    CM_CONST = 1 << 3, CM_VAR = 2 << 3, CM_FUN = 3 << 3
};

typedef struct cm_variable {
    const char *name;
    const void *value;
    int type;
} cm_variable;

// Optimization flags
#define CM_OPT_NONE          0x00
#define CM_OPT_PATTERN       0x01  // Pattern specialized
#define CM_OPT_BYTECODE      0x02  // Bytecode compiled
#define CM_OPT_JIT           0x04  // JIT compiled
#define CM_OPT_SIMD          0x08  // SIMD optimized
#define CM_OPT_CONST_FOLDED  0x10  // Constants folded
#define CM_OPT_SPECIALIZED   0x20  // Expression specialized

// Parses the input expression, evaluates it and frees it
double cm_interp(const char *expression, int *error);

// Parses the input expression and binds variables
cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error);

// Evaluate the expression
double cm_eval(const cm_expr *n, int *error);

// Prints debugging information
void cm_print(const cm_expr *n);

// Frees expression
void cm_free(cm_expr *n);

#ifdef __cplusplus
}
#endif
#endif //__CMATH_H__
