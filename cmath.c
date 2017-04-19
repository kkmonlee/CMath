//
// Created by aa on 19/12/16.
//

// Test commit by Lewis

#include "cmath.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

enum {TOK_NULL, TOK_END, TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_INFIX,
    TOK_VARIABLE, TOK_SEP, TOK_ERROR, TOK_FUNCTION0, TOK_FUNCTION1,
    TOK_FUNCTION2, TOK_FUNCTION3, TOK_FUNCTION4, TOK_FUNCTION5, TOK_FUNCTION6,
    TOK_FUNCTION7
};

typedef struct state {
    const char *start;
    const char *next;
    int type;
    union {
        double value;
        cm_fun fun;
        const double *bound;
    };

    const cm_variable *lookup;
    int lookup_len;
} state;

static int cm_get_type(const int type) {
    if (type == 0) return CM_VAR;
    return type & CM_FLAG_TYPE;
}

static int cm_get_arity(const int type) {
    return type & CM_MASK_ARIT;
}

static cm_expr *new_expr(const int type, const cm_expr *members[]) {
    int member_count = cm_get_arity(type);
    size_t member_size = sizeof(cm_expr*) *member_count;
    cm_expr *ret = malloc(sizeof(cm_expr) + member_size);
    if (!members) {
        memset(ret->members, 0, member_size);
    } else {
        memcpy(ret->members, members, member_size);
    }
    ret->type = type;
    ret->bound = 0;
    return ret;
}

void cm_free(cm_expr *n) {
    int i;
    if (!n) return;
    for (i = n->member_count - 1; i >= 0; i--) {
        cm_free(n->members[i]);
    }
    free(n);
}

static const double pi = 3.14159265358979323846;
static const double e = 2.71828182845904523536;

static const cm_variable functions[] = {
        {"abs", fabs,     CM_FUN | 1},
        {"acos", acos,    CM_FUN | 1},
        {"asin", asin,    CM_FUN | 1},
        {"atan", atan,    CM_FUN | 1},
        {"atan2", atan2,  CM_FUN | 2},
        {"ceil", ceil,    CM_FUN | 1},
        {"cos", cos,      CM_FUN | 1},
        {"cosh", cosh,    CM_FUN | 1},
        {"e", &e,         CM_VAR},
        {"exp", exp,      CM_FUN | 1},
        {"floor", floor,  CM_FUN | 1},
        {"ln", log,       CM_FUN | 1},
        {"log", log10,    CM_FUN | 1},
        {"pi", &pi,       CM_VAR},
        {"pow", pow,      CM_FUN | 2},
        {"sin", sin,      CM_FUN | 1},
        {"sinh", sinh,    CM_FUN | 1},
        {"sqrt", sqrt,    CM_FUN | 1},
        {"tan", tan,      CM_FUN | 1},
        {"tanh", tanh,    CM_FUN | 1},
        {0}

};

static const cm_variable *find_function(const char *name, int len) {
    int imin = 0;
    int imax = sizeof(functions) / sizeof(cm_variable) - 2;

    while (imax >= imin) {
        const int i = (imin + ((imax - imin) / 2));
        int c = strncmp(name, functions[i].name, (size_t) len);
        if (!c)
            c = '\0' - functions[i].name[len];
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

static const cm_variable *find_lookup(const state *s, const char *name, int len) {
    int i;
    if (!s->lookup) return 0;
    for (i = 0; i < s->lookup_len; ++i) {
        if (strncmp(name, s->lookup[i].name, len) == 0 && s->lookup[i].name[len] == '\0') {
            return s->lookup + i;
        }
    }

    return 0;
}

static double add(double a, double b) {return a + b;};
static double sub(double a, double b) {return a - b;}
static double mul(double a, double b) {return a * b;}
static double divide(double a, double b) {return a / b;}
static double negate(double a) {return -a;}
static double comma(double a, double b) {return b;}

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
                int arity, type;
                const char *start;
                start = s->next;
                while ((s->next[0] >= 'a' && s->next[0] <= 'z') || (s->next[0] >= '0' && s->next[0] <= '9'))
                    s->next++;
                const cm_variable *var = find_lookup(s, start, s->next - start);
                if (!var)
                    var = find_function(start, s->next - start);
                if (!var) {
                    s->type = TOK_ERROR;
                } else {
                    type = cm_get_type(var->type);
                    arity = cm_get_arity(var->type);
                    switch (type) {
                        case CM_VAR:
                            s->type = TOK_VARIABLE;
                            s->bound = var->value;
                            break;
                        case CM_FUN:
                            s->type = TOK_FUNCTION0 + arity;
                            s->fun.f0 = (void*) var->value;
                    }
                }
            } else {
                switch (s->next++[0]) {
                    case '+': s->type = TOK_INFIX; s->fun.f2 = add; break;
                    case '-': s->type = TOK_INFIX; s->fun.f2 = sub; break;
                    case '*': s->type = TOK_INFIX; s->fun.f2 = mul; break;
                    case '/': s->type = TOK_INFIX; s->fun.f2 = divide; break;
                    case '^': s->type = TOK_INFIX; s->fun.f2 = pow; break;
                    case '%': s->type = TOK_INFIX; s->fun.f2 = fmod; break;
                    case '(': s->type = TOK_OPEN; break;
                    case ')': s->type = TOK_CLOSE; break;
                    case ',': s->type = TOK_SEP; break;
                    case ' ': case '\t': case '\n': case '\r': break;
                    default: s->type = TOK_ERROR; break;
                }
            }
        }
    } while (s->type == TOK_NULL);
}

static cm_expr *list(state *s);
static cm_expr *expr(state *s);
static cm_expr *power(state *s);

static cm_expr *base(state *s) {
    cm_expr *ret;
    int arity;

    switch (s->type) {
        case TOK_NUMBER:
            ret = new_expr(CM_CONST, 0);
            ret->value = s->value;
            next_token(s);
            break;

        case TOK_VARIABLE:
            ret = new_expr(CM_VAR, 0);
            ret->bound = s->bound;
            next_token(s);
            break;

        case TOK_FUNCTION0:
            ret = new_expr(CM_FUN, 0);
            ret = new_expr(CM_FUN, 0);
            ret->fun.f0 = s->fun.f0;
            next_token(s);
            if (s->type == TOK_OPEN) {
                next_token(s);
                if (s->type != TOK_CLOSE) {
                    s->type = TOK_ERROR;
                } else {
                    next_token(s);
                }
            }
            break;

        case TOK_FUNCTION1:
            ret = new_expr(CM_FUN | 1, 0);
            ret->fun.f0 = s->fun.f0;
            next_token(s);
            ret->members[0] = power(s);
            break;

        case TOK_FUNCTION2: case TOK_FUNCTION3: case TOK_FUNCTION4:
        case TOK_FUNCTION5: case TOK_FUNCTION6: case TOK_FUNCTION7:
            arity = s->type - TOK_FUNCTION0;

            ret = new_expr(CM_FUN | arity, 0);
            ret->fun.f0 = s->fun.f0;
            next_token(s);

            if (s->type != TOK_OPEN) {
                s->type = TOK_ERROR;
            } else {
                int i;
                for(i = 0; i < arity; i++) {
                    next_token(s);
                    ret->members[i] = expr(s);
                    if(s->type != TOK_SEP) {
                        break;
                    }
                }
                if(s->type != TOK_CLOSE || i < arity - 1) {
                    s->type = TOK_ERROR;
                } else {
                    next_token(s);
                }
            }

            break;

        case TOK_OPEN:
            next_token(s);
            ret = list(s);
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
    while (s->type == TOK_INFIX && (s->fun.f2 == add || s->fun.f2 == sub)) {
        if (s->fun.f2 == sub) {
            sign = -sign;
        }
        next_token(s);
    }

    cm_expr *ret;

    if (sign == 1) ret = base(s);
    else {
        ret = new_expr(CM_FUN | 1, (const cm_expr *[]) {
                base(s)
        });
        ret->fun.f1 = negate;
    }

    return ret;
}

static cm_expr *factor(state *s) {
    cm_expr *ret = power(s);

    while (s->type == TOK_INFIX && (s->fun.f2 == pow)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, power(s)
        });
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *term(state *s) {
    cm_expr *ret = factor(s);

    while (s->type == TOK_INFIX && (s->fun.f2 == mul || s->fun.f2 == divide || s->fun.f2 == fmod)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, factor(s)
        });
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *expr(state *s) {
    cm_expr *ret = term(s);

    while (s->type == TOK_INFIX && (s->fun.f2 == add || s->fun.f2 == sub)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, term(s)
        });
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *list(state *s) {
    cm_expr *ret = expr(s);

    while (s->type == TOK_SEP) {
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, term(s)
        });
        ret->fun.f2 = comma;
    }
    return ret;
}

double cm_eval(const cm_expr *n) {
    switch(cm_get_type(n->type)) {
        case CM_CONST: return n->value;
        case CM_VAR: return *n->bound;
        case CM_FUN:
            switch(cm_get_arity(n->type)) {
#define m(e) cm_eval(n->members[e])
                case 0: return n->fun.f0();
                case 1: return n->fun.f1(m(0));
                case 2: return n->fun.f2(m(0), m(1));
                case 3: return n->fun.f3(m(0), m(1), m(2));
                case 4: return n->fun.f4(m(0), m(1), m(2), m(3));
                case 5: return n->fun.f5(m(0), m(1), m(2), m(3), m(4));
                case 6: return n->fun.f6(m(0), m(1), m(2), m(3), m(4), m(5));
                case 7: return n->fun.f7(m(0), m(1), m(2), m(3), m(4), m(5), m(6));
                default: return 0.0/0.0;
#undef m
            }
        default: return 0.0 / 0.0;
    }
}

static void optimise(cm_expr *n) {
/*    if (n->bound) return;

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
    }*/
}

cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error) {
    state s;
    s.start = s.start = expression;
    s.lookup = variables;
    s.lookup_len = var_count;

    next_token(&s);
    cm_expr *root = list(&s);

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
    int i, arity;
    printf("%*s", depth, "");

    switch (cm_get_type(n->type)) {
        case CM_CONST:
            printf("%f\n", n->value);
            break;
        case CM_VAR:
            printf("bound %p\n", n->bound);
            break;
        case CM_FUN:
            arity = cm_get_arity(n->type);
            printf("f%d", arity);
            for (i = 0; i < arity; i++) {
                printf(" %p", n->members[i]);
            }
            printf("\n");
            for (i = 0; i < arity; i++) {
                pn(n->members[i], depth + 1);
            }
            break;
    }
}

void cm_print(const cm_expr *n) {
    pn(n, 0);
}
