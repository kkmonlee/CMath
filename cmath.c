//
// Created by aa on 19/12/16.
//

// Test commit by Lewis

#include "cmath.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>

enum {
    TOK_NULL, TOK_END, TOK_OPEN, TOK_CLOSE, TOK_NUMBER, TOK_INFIX,
    TOK_VARIABLE, TOK_SEP, TOK_ERROR, TOK_FUNCTION0, TOK_FUNCTION1,
    TOK_FUNCTION2, TOK_FUNCTION3, TOK_FUNCTION4, TOK_FUNCTION5, TOK_FUNCTION6,
    TOK_FUNCTION7
};

typedef enum {
    CM_ERROR_NONE,
    CM_ERROR_SYNTAX,
    CM_ERROR_DIVISION_BY_ZERO,
    CM_ERROR_UNDEFINED_VARIABLE,
    CM_ERROR_INVALID_FUNCTION_CALL
} cm_error_code;

#define POOL_SIZE 1024
typedef struct {
    cm_expr *nodes[POOL_SIZE];
    int nextFree; // index of the next free node
    pthread_mutex_t mutex;
} cm_expr_pool;

cm_expr_pool globalPool = { {0}, 0, PTHREAD_MUTEX_INITIALIZER };

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

void cm_init_pool(cm_expr_pool *pool) {
    pthread_mutex_init(&pool->mutex, NULL);
    pool->nextFree = 0;
}

// Function to compact memory pool
void cm_compact_pool(cm_expr_pool *pool) {
    int dest = 0; // destination index for compaction

    for (int src = 0; src < pool->nextFree; ++src) {
        cm_expr *node = &pool->nodes[src];

        // if the node is in use and not at its correct position
        if (node->member_count >= 0 && src != dest) {
            // move the node and its members to the new location
            memmove(&pool->nodes[dest], 
                node,
                sizeof(cm_expr) + sizeof(cm_expr*) * node->member_count
            );

            // update pointers in parent nodes
            for (int i = 0; i < pool->nextFree; ++i) {
                cm_expr *parent = &pool->nodes[i];
                for (int j = 0; j < parent->member_count; ++j) {
                    if (parent->members[j] == node) {
                        parent->members[j] = &pool->nodes[dest];
                    }
                }
            }
        }

        // if the node was in use, move the destination index
        if (node->member_count >= 0) {
            dest += node->member_count + 1;
        }
    }
    pool->nextFree = dest;
}

cm_expr *cm_pool_alloc(cm_expr_pool *pool, int member_count) {
    assert(member_count >= 0);
    pthread_mutex_lock(&pool->mutex); // lock mutex

    // try to allocate without compaction first
    if (pool->nextFree + member_count < POOL_SIZE) {
        return new_expr(0, NULL, pool);
    }

    // compaction failed, so compact the pool
    cm_compact_pool(pool);

    if (pool->nextFree + member_count < POOL_SIZE) {
        pthread_mutex_unlock(&pool->mutex);
        return new_expr(0, NULL, pool);
    } else {
        fprintf(stderr, "Expression pool exhausted\n");
        pthread_mutex_unlock(&pool->mutex);
        exit(1);
    }
}

// Reset the pool to the initial state
void cm_reset_pool(cm_expr_pool *pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->nextFree = 0;
    pthread_mutex_unlock(&pool->mutex);
}

static int cm_get_type(const int type) {
    if (type == 0) return CM_VAR;
    return type & CM_FLAG_TYPE;
}

static int cm_get_arity(const int type) {
    return type & CM_MASK_ARIT;
}

static cm_expr *new_expr(const int type, const cm_expr *members[], cm_expr_pool *pool) {
    int member_count = cm_get_arity(type);
    cm_expr *ret = cm_pool_alloc(pool, member_count);
    ret->type = type;
    if (members) {
        memcpy(ret->members, members, sizeof(cm_expr*) * member_count);
    }

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

    if (s->type == TOK_ERROR) {
        s->next--;
        return;
    }
}

static cm_expr *list(state *s, cm_error_code *error);
static cm_expr *expr(state *s, cm_error_code *error);
static cm_expr *power(state *s, cm_error_code *error);
static cm_expr *factor(state *s, cm_error_code *error);
static cm_expr *term(state *s, cm_error_code *error);

static cm_expr *base(state *s, cm_error_code *error) {
    cm_expr *ret;
    int arity;

    if (s->type == TOK_VARIABLE) {
        if (!s->bound) {
            *error = CM_ERROR_UNDEFINED_VARIABLE;
            return NULL;
        }
    }

    switch (s->type) {
        case TOK_NUMBER:
            ret = new_expr(CM_CONST, 0, &globalPool);
            ret->value = s->value;
            next_token(s);
            break;

        case TOK_VARIABLE:
            ret = new_expr(CM_VAR, 0, &globalPool);
            ret->bound = s->bound;
            next_token(s);
            break;

        case TOK_FUNCTION0:
            ret = new_expr(CM_FUN, 0, &globalPool);
            ret = new_expr(CM_FUN, 0, &globalPool);
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
            ret = new_expr(CM_FUN | 1, 0, &globalPool);
            ret->fun.f0 = s->fun.f0;
            next_token(s);
            ret->members[0] = power(s, error);
            break;

        case TOK_FUNCTION2: case TOK_FUNCTION3: case TOK_FUNCTION4:
        case TOK_FUNCTION5: case TOK_FUNCTION6: case TOK_FUNCTION7:
            arity = s->type - TOK_FUNCTION0;

            ret = new_expr(CM_FUN | arity, 0, &globalPool);
            ret->fun.f0 = s->fun.f0;
            next_token(s);

            if (s->type != TOK_OPEN) {
                s->type = TOK_ERROR;
            } else {
                int i;
                for(i = 0; i < arity; i++) {
                    next_token(s);
                    ret->members[i] = expr(s, error);
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
            ret = list(s, error);
            if (s->type != TOK_CLOSE) {
                s->type = TOK_ERROR;
            } else {
                next_token(s);
            }
            break;

        default:
            ret = new_expr(0, 0, &globalPool);
            s->type = TOK_ERROR;
            ret->value = 0.0 / 0.0;
            break;
    }

    return ret;
}

static cm_expr *power(state *s, cm_error_code *error) {
    int sign = 1;
    while (s->type == TOK_INFIX && (s->fun.f2 == add || s->fun.f2 == sub)) {
        if (s->fun.f2 == sub) {
            sign = -sign;
        }
        next_token(s);
    }

    cm_expr *ret;

    if (sign == 1) ret = base(s, error);
    else {
        ret = new_expr(CM_FUN | 1, (const cm_expr *[]) {
                base(s, error)
        }, &globalPool);
        ret->fun.f1 = negate;
    }

    return ret;
}

static cm_expr *factor(state *s, cm_error_code *error) {
    cm_expr *ret = power(s, error);

    while (s->type == TOK_INFIX && (s->fun.f2 == pow)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, power(s, error)
        }, &globalPool);
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *term(state *s, cm_error_code *error) {
    cm_expr *ret = factor(s, error);

    while (s->type == TOK_INFIX && (s->fun.f2 == mul || s->fun.f2 == divide || s->fun.f2 == fmod)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, factor(s, error)
        }, &globalPool);
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *expr(state *s, cm_error_code *error) {
    cm_expr *ret = term(s, error);

    while (s->type == TOK_INFIX && (s->fun.f2 == add || s->fun.f2 == sub)) {
        cm_fun2 t = s->fun.f2;
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, term(s, error)
        }, &globalPool);
        ret->fun.f2 = t;
    }

    return ret;
}

static cm_expr *list(state *s, cm_error_code *error) {
    cm_expr *ret = expr(s, error);

    while (s->type == TOK_SEP) {
        next_token(s);
        ret = new_expr(CM_FUN | 2, (const cm_expr *[]) {
                ret, term(s, error)
        }, &globalPool);
        ret->fun.f2 = comma;
    }
    return ret;
}

double cm_eval(const cm_expr *n, cm_error_code *error) {
    switch(cm_get_type(n->type)) {
        case CM_CONST: return n->value;
        case CM_VAR: return *n->bound;
        case CM_FUN:
            switch(cm_get_arity(n->type)) {
#define m(e) cm_eval(n->members[e], error)
                case 0: return n->fun.f0();
                case 1: return n->fun.f1(m(0));
                case 2:
                    if (n->fun.f2 == divide && cm_eval(n->members[1], error) == 0) {
                        *error = CM_ERROR_DIVISION_BY_ZERO;
                        return 0.0 / 0.0; // NaN
                    }
                    return n->fun.f2(m(0), m(1));
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
    *error = CM_ERROR_SYNTAX;
    return 0.0 / 0.0;
}

static void optimise(cm_expr *n) {
    if (!n) return;

    for (int i = 0; i < n->member_count; ++i) {
        optimise(n->members[i]);
    }

    // constant folding (simple case)
    if (cm_get_type(n->type) == CM_FUN && n->member_count == 2) {
        if (cm_get_type(n->members[0]->type) == CM_CONST &&
            cm_get_type(n->members[1]->type) == CM_CONST) {
            n->type = CM_CONST;
            n->value = n->fun.f2(n->members[0]->value, n->members[1]->value);
            cm_free(n->members[0]);
            cm_free(n->members[1]);
            n->member_count = 0;
        }
    }
}

cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error) {
    cm_init_pool(&globalPool);
    state s;
    s.start = s.start = expression;
    s.lookup = variables;
    s.lookup_len = var_count;

    cm_error_code parse_error = CM_ERROR_NONE;
    next_token(&s);
    cm_expr *root = list(&s, parse_error);

    if (parse_error != CM_ERROR_NONE) {
        cm_free(root);
        if (error) {
            *error = (int) (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        return NULL;
    } else {
        optimise(root);
        if (error) *error = 0;
        return root;
    }
}

double cm_interp(const char *expression, cm_error_code *error) {
    cm_expr *n = cm_compile(expression, 0, 0, error);
    double ret;
    if (n) {
        ret = cm_eval(n, error);
        cm_free(n);
    } else {
        ret = 0.0 / 0.0;
    }

    return ret;
}

static void pn(const cm_expr *n, int depth) {
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
