//
// Created by aa on 19/12/16.
//

// Test commit by Lewis

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
    if (!n) return;
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
static double mod(double a, double b) {
    if (((long long) b) == 0) return 0.0 / 0.0;
    return ((long long) a % (long long) b);
}
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
            ret->value = 0.0 / 0.0;
            break;
    }

    return ret;
}

static cm_expr *power(state *s) {
    int sign = 1;
    while (s->type == TOK_FUNCTION2 && (s->f2 == add || s->f2 == sub)) {
        if (s->f2 == sub) {
            sign = -sign;
        }
        next_token(s);
    }

    cm_expr *ret;

    if (sign == 1) ret = base(s);
    else {
        ret = new_expr(base(s), 0);
        ret->f1 = negate;
    }

    return ret;
}

static cm_expr *factor(state *s) {
    cm_expr *ret = power(s);

    while (s->type == TOK_FUNCTION2 && (s->f2 == pow)) {
        cm_fun2 t = s->f2;
        next_token(s);
        ret = new_expr(ret, power(s));
        ret->f2 = t;
    }

    return ret;
}

static cm_expr *term(state *s) {
    cm_expr *ret = factor(s);

    while (s->type == TOK_FUNCTION2 && (s->f2 == mul || s->f2 == divide || s->f2 == mod)) {
        cm_fun2 t = s->f2;
        next_token(s);
        ret = new_expr(ret, factor(s));
        ret->f2 = t;
    }

    return ret;
}

static cm_expr *expr(state *s) {
    cm_expr *ret = term(s);

    while (s->type == TOK_FUNCTION2 && (s->f2 == add || s->f2 == sub)) {
        cm_fun2 t = s->f2;
        next_token(s);
        ret = new_expr(ret, term(s));
        ret->f2 = t;
    }

    return ret;
}

double cm_eval(const cm_expr *n) {
    double ret;

    if (n->bound) {
        ret = *n->bound;
    } else if (n->left == 0 && n->right == 0) {
        ret = n->value;
    } else if (n->left && n->right == 0) {
        ret = n->f1(cm_eval(n->left));
    } else {
        ret = n->f2(cm_eval(n->left), cm_eval(n->right));
    }

    return ret;
}

static void optimise(cm_expr *n) {
    if (n->bound) return;

    if (n->left) optimise(n->left);
    if (n->right) optimise(n->right);

    if (n->left && n->right) {
        if (n->left->left == 0 && n->left->right == 0 && n->right->left == 0 && n->right->right == 0 && n->right->bound == 0 && n->left->bound == 0) {
            const double r = n->f2(n->left->value, n->right->value);
            free(n->left);
            free(n->right);
            n->value = r;
        }
    } else if (n->left && !n->right) {
        if (n->left->left == 0 && n->left->right == 0 && n->left->bound == 0) {
            const double r = n->f1(n->left->value);
            free(n->left);
            n->left = 0;
            n->value = r;
        }
    }
}

cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error) {
    state s;
    s.start = s.start = expression;
    s.lookup = variables;
    s.lookup_len = var_count;

    next_token(&s);
    cm_expr *root = expr(&s);

    if (s.type != TOK_END) {
        cm_free(root);
        if (error) {
            *error = (int) (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        return 0;
    } else {
        optimise(root);
        if (error) *error = 0;
        return root;
    }
}

double cm_interp(const char *expression, int *error) {
    cm_expr *n = cm_compile(expression, 0, 0, error);
    double ret;
    if (n) {
        ret = cm_eval(n);
        cm_free(n);
    } else {
        ret = 0.0 / 0.0;
    }

    return ret;
}

static void pn (const cm_expr *n, int depth) {
    printf("%*s", depth, "");

    if (n->bound) {
        printf("bound %p\n", n->bound);
    } else if (n->left == 0 && n->right == 0) {
        printf("%f\n", n->value);
    } else if (n->left && n->right == 0) {
        printf("f1 %p\n", n->left);
    } else {
        printf("f2 %p %p\n", n->left, n->right);
        pn(n->left, depth + 1);
        pn(n->right, depth + 1);
    }
}

void cm_print(const cm_expr *n) {
    pn(n, 0);
}
