//
// Created by aa on 19/12/16.
//

#include "cmath.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

enum {TOK_NULL, TOK_END, TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_ADD, TOK_SUB, TOK_MUL, TOK_DIV, TOK_FUNCTION1, TOK_FUNCTION2, TOK_VARIABLE, TOK_ERROR};

typedef struct {
    const char *start;
    const char *next;
    int type;
    union {double value; cm_fun1 f1; cm_fun2 f2; const double *var;};

    const cm_variable *lookup;
    int lookup_len;
} state;

static cm_expr *new_expr(cm_expr *l, cm_expr *r) {
    cm_expr *ret = malloc(sizeof(cm_expr));
    ret->left = l;
    ret->right = r;
    ret->bound = 0;
    return ret;
}

void cm_free(cm_expr *n) {
    if (n->left) cm_free(n->left);
    if (n->right) cm_free(n->right);
    free(n);
}

typedef struct {
    const char *name;
    cm_fun1 f1;
} builtin;

static const builtin functions[] = {
        {"abs", fabs},
        {"acos", acos},
        {"asin", asin},
        {"atan", atan},
        {"ceil", ceil},
        {"cos", cos},
        {"cosh", cosh},
        {"exp", exp},
        {"floor", floor},
        {"ln", log},
        {"log", log10},
        {"sin", sin},
        {"sinh", sinh},
        {"sqrt", sqrt},
        {"tan", tan},
        {"tanh", tanh},
        {0}
};

static const builtin *find_function(const char *name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(builtin) - 2;

    while (imax >= imin) {
        const int i = (imin + ((imax - imin) / 2));
        int c = strncmp(name, functions[i].name, (size_t) len);
        if (!c) c = (int) (len - strlen(functions[i].name));
        if (c == 0) {
            return functions + i;
        } else if (c > 0) {
            imin = i + 1;
        } else {
            imax = i - 1;
        }
    }

    return 0;
}

static const double *find_var(const state *s, const char *name, int len) {
    int i;
    if (!s->lookup) return 0;
    for (i = 0; i < s->lookup_len; ++i) {
        if (strlen(s->lookup[i].name) == len && strncmp(name, s->lookup[i].name, len) == 0) {
            return s->lookup[i].value;
        }
    }

    return 0;
}

static double add(double a, double b) {return a + b;};
static double sub(double a, double b) {return a - b;}
static double mul(double a, double b) {return a * b;}
static double divide(double a, double b) {return a / b;}
static double mod(double a, double b) {return (long long)a % (long long)b;}
static double negate(double a) {return -a;}

void next_token(state *s) {
    s->type = TOK_NULL;

    if (!*s->next) {
        s->type = TOK_END;
        return;
    }

    do {
        if ((s->next[0] >= '0' && s->next[0] <= '9') || s->next[0] == '.') {
            s->value = strtod(s->next, (char**)&s->next);
            s->type = TOK_NUMBER;
        } else {
            if (s->next[0] >= 'a' && s->next[0] <= 'z') {
                const char *start;
                start = s->next;
                while (s->next[0] >= 'a' && s->next[0] <= 'z') s->next++;

                const double *var = find_var(s, start, (s->next - start));
                if (var) {
                    s->type = TOK_VARIABLE;
                    s->var = var;
                } else {
                    if (s->next - start > 15) {
                        s->type = TOK_ERROR;
                    } else {
                        s->type = TOK_FUNCTION1;
                        const builtin *f = find_function(start, s->next - start);
                        if (!f) {
                            s->type = TOK_ERROR;
                        } else {
                            s->f1 = f->f1;
                        }
                    }
                }
            } else {
                switch (s->next++[0]) {
                    case '+': s->type = TOK_FUNCTION2; s->f2 = add; break;
                    case '-': s->type = TOK_FUNCTION2; s->f2 = sub; break;
                    case '*': s->type = TOK_FUNCTION2; s->f2 = mul; break;
                    case '/': s->type = TOK_FUNCTION2; s->f2 = divide; break;
                    case '^': s->type = TOK_FUNCTION2; s->f2 = pow; break;
                    case '%': s->type = TOK_FUNCTION2; s->f2 = mod; break;
                    case '(': s->type = TOK_OPEN; break;
                    case ')': s->type = TOK_CLOSE; break;
                    case ' ': case '\t': case '\n': case '\r': break;
                    default: s->type = TOK_ERROR; break;
                }
            }
        }
    } while (s->type == TOK_NULL);
}

static cm_expr *expr(state *s);
static cm_expr *power(state *s);

static cm_expr *base(state *s) {
    cm_expr *ret;

    switch (s->type) {
        case TOK_NUMBER:
            ret = new_expr(0, 0);
            ret->value = s->value;
            next_token(s);
            break;

        case TOK_VARIABLE:
            ret = new_expr(0, 0);
            ret->bound = s->var;
            next_token(s);
            break;

        case TOK_FUNCTION1:
            ret = new_expr(0, 0);
            ret->f1 = s->f1;
            next_token(s);
            ret->left = power(s);
            break;

        case TOK_OPEN:
            next_token(s);
            ret - expr(s);
            if (s->type != TOK_CLOSE) {
                s->type = TOK_ERROR;
            } else {
                next_token(s);
            }
            break;

        default:
            ret = new_expr(0, 0);
            s->type = TOK_ERROR;
            ret->value = 1.0 / 0.0;
            break;
    }

    return ret;
}