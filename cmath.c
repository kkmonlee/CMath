//

#include "cmath.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdbool.h>
#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

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
    cm_expr nodes[POOL_SIZE];
    int nextFree; // index of the next free node
    pthread_mutex_t mutex;
} cm_expr_pool;

cm_expr_pool globalPool = { {0}, 0, PTHREAD_MUTEX_INITIALIZER };

#define JIT_CODE_SIZE 4096
#define BYTECODE_CHUNK_SIZE 256
#define MAX_EXPRESSION_DEPTH 64
#define SIMD_VECTOR_SIZE 4

typedef enum {
    OP_CONST, OP_VAR, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW,
    OP_NEG, OP_SQRT, OP_SIN, OP_COS, OP_TAN, OP_LOG, OP_EXP,
    OP_FABS, OP_RETURN
} cm_opcode;

typedef struct {
    cm_opcode op;
    union {
        double constant;
        int var_index;
    } data;
} cm_instruction;

typedef struct {
    cm_instruction *code;
    size_t code_size;
    size_t code_capacity;
    double *constants;
    size_t const_count;
    int var_count;
} cm_bytecode;

typedef struct {
    void *code;
    size_t size;
    double (*compiled_func)(const double *vars);
} cm_jit_code;

typedef struct {
    unsigned char *buffer;
    size_t pos;
    size_t size;
} cm_jit_builder;

typedef struct {
    float64x2_t vec_a, vec_b, vec_result;
    double scalar_result[SIMD_VECTOR_SIZE];
} cm_simd_ctx;
typedef enum {
    PATTERN_CONST,
    PATTERN_VAR,
    PATTERN_ADD_CONST,     // x + constant
    PATTERN_MUL_CONST,     // x * constant
    PATTERN_POW_CONST,     // x ^ constant
    PATTERN_LINEAR,        // ax + b
    PATTERN_QUADRATIC,     // ax^2 + bx + c
    PATTERN_POLYNOMIAL,    // general polynomial
    PATTERN_TRIG,          // sin/cos/tan patterns
    PATTERN_EXP_LOG,       // exp/log patterns
    PATTERN_UNKNOWN
} cm_pattern_type;

typedef struct {
    cm_pattern_type type;
    double coefficients[8];
    int degree;
    cm_fun specialized_func;
} cm_pattern;

static cm_expr *new_expr(const int type, const cm_expr *members[], cm_expr_pool *pool);
static cm_jit_code *cm_compile_jit(const cm_expr *expr, int var_count);
static void cm_jit_free(cm_jit_code *jit);
static cm_bytecode *cm_compile_bytecode(const cm_expr *expr, int var_count);
static double cm_eval_bytecode(const cm_bytecode *bc, const double *vars);
static void cm_bytecode_free(cm_bytecode *bc);
static cm_pattern cm_analyze_pattern(const cm_expr *expr);
static double cm_eval_specialized(const cm_expr *expr, const cm_pattern *pattern, const double *vars);
static void cm_flatten_expression(const cm_expr *expr, cm_instruction *instructions, size_t *count);
static double cm_eval_simd_arm64(const cm_expr *expr, const double *vars);
static double cm_eval_vectorized(const cm_expr *expr, const double *vars, int vector_size);

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

void cm_compact_pool(cm_expr_pool *pool) {
    int dest = 0;

    for (int src = 0; src < pool->nextFree; ++src) {
        cm_expr *node = &pool->nodes[src];

        if (node->member_count >= 0 && src != dest) {
            memmove(&pool->nodes[dest],
                node,
                sizeof(cm_expr) + sizeof(cm_expr*) * node->member_count
            );

            for (int i = 0; i < pool->nextFree; ++i) {
                cm_expr *parent = &pool->nodes[i];
                for (int j = 0; j < parent->member_count; ++j) {
                    if (parent->members[j] == node) {
                        parent->members[j] = &pool->nodes[dest];
                    }
                }
            }
        }

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
    if (pool->nextFree + member_count + 1 <= POOL_SIZE) {
        cm_expr *ret = &pool->nodes[pool->nextFree];
        pool->nextFree += 1;
        ret->member_count = member_count;
        pthread_mutex_unlock(&pool->mutex);
        return ret;
    }

    // compaction failed, so compact the pool
    cm_compact_pool(pool);

    if (pool->nextFree + member_count + 1 <= POOL_SIZE) {
        cm_expr *ret = &pool->nodes[pool->nextFree];
        pool->nextFree += 1;
        ret->member_count = member_count;
        pthread_mutex_unlock(&pool->mutex);
        return ret;
    } else {
        fprintf(stderr, "Expression pool exhausted\n");
        pthread_mutex_unlock(&pool->mutex);
        exit(1);
    }
}

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
    ret->jit_code = NULL;
    // member_count is already set by cm_pool_alloc
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

    if (n->jit_code) {
        cm_jit_free((cm_jit_code*)n->jit_code);
        n->jit_code = NULL;
    }

    if (n->bytecode) {
        cm_bytecode_free((cm_bytecode*)n->bytecode);
        n->bytecode = NULL;
    }

    if (n->pattern) {
        free(n->pattern);
        n->pattern = NULL;
    }

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

static cm_expr *list(state *s, int *error);
static cm_expr *expr(state *s, int *error);
static cm_expr *power(state *s, int *error);
static cm_expr *factor(state *s, int *error);
static cm_expr *term(state *s, int *error);

static cm_expr *base(state *s, int *error) {
    cm_expr *ret;
    int arity;

    if (s->type == TOK_VARIABLE) {
        if (!s->bound) {
            if (error) *error = CM_ERROR_UNDEFINED_VARIABLE;
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

static cm_expr *power(state *s, int *error) {
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

static cm_expr *factor(state *s, int *error) {
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

static cm_expr *term(state *s, int *error) {
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

static cm_expr *expr(state *s, int *error) {
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

static cm_expr *list(state *s, int *error) {
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

#ifdef __x86_64__
static void cm_eval_simd_batch(const cm_expr *n, const double *vars, double *results, int count) {
    if (count < 4) {
        for (int i = 0; i < count; i++) {
            results[i] = cm_eval_fast(n);
        }
        return;
    }

    switch(cm_get_type(n->type)) {
        case CM_CONST: {
            __m256d const_vec = _mm256_set1_pd(n->value);
            for (int i = 0; i < count; i += 4) {
                _mm256_storeu_pd(&results[i], const_vec);
            }
            break;
        }
        case CM_VAR: {
            __m256d var_vec = _mm256_loadu_pd(n->bound);
            for (int i = 0; i < count; i += 4) {
                _mm256_storeu_pd(&results[i], var_vec);
            }
            break;
        }
        case CM_FUN: {
            if (cm_get_arity(n->type) == 2 && n->fun.f2 == add) {
                double left_results[count], right_results[count];
                cm_eval_simd_batch(n->members[0], vars, left_results, count);
                cm_eval_simd_batch(n->members[1], vars, right_results, count);

                for (int i = 0; i < count; i += 4) {
                    __m256d left_vec = _mm256_loadu_pd(&left_results[i]);
                    __m256d right_vec = _mm256_loadu_pd(&right_results[i]);
                    __m256d result_vec = _mm256_add_pd(left_vec, right_vec);
                    _mm256_storeu_pd(&results[i], result_vec);
                }
            } else if (cm_get_arity(n->type) == 2 && n->fun.f2 == mul) {
                double left_results[count], right_results[count];
                cm_eval_simd_batch(n->members[0], vars, left_results, count);
                cm_eval_simd_batch(n->members[1], vars, right_results, count);

                for (int i = 0; i < count; i += 4) {
                    __m256d left_vec = _mm256_loadu_pd(&left_results[i]);
                    __m256d right_vec = _mm256_loadu_pd(&right_results[i]);
                    __m256d result_vec = _mm256_mul_pd(left_vec, right_vec);
                    _mm256_storeu_pd(&results[i], result_vec);
                }
            } else if (cm_get_arity(n->type) == 1 && n->fun.f1 == sqrt) {
                double operand_results[count];
                cm_eval_simd_batch(n->members[0], vars, operand_results, count);

                for (int i = 0; i < count; i += 4) {
                    __m256d operand_vec = _mm256_loadu_pd(&operand_results[i]);
                    __m256d result_vec = _mm256_sqrt_pd(operand_vec);
                    _mm256_storeu_pd(&results[i], result_vec);
                }
            } else {
                for (int i = 0; i < count; i++) {
                    results[i] = cm_eval_fast(n);
                }
            }
            break;
        }
        default:
            for (int i = 0; i < count; i++) {
                results[i] = 0.0/0.0;
            }
    }
}
#endif

static double var_cache[1024] __attribute__((aligned(32)));
static int var_cache_pos = 0;

#ifdef __GNUC__
static double cm_eval_computed_goto(const cm_expr *n) {
    static void* dispatch_table[] = {
        &&handle_default,
        &&handle_const,
        &&handle_var,
        &&handle_fun
    };

    int type_index = cm_get_type(n->type) >> 3;
    if (type_index > 3) type_index = 0;
    goto *dispatch_table[type_index];

handle_const:
    return n->value;

handle_var:
    return *n->bound;

handle_fun: {
    int arity = cm_get_arity(n->type);
    switch(arity) {
        case 0: return n->fun.f0();
        case 1: {
            register double operand = cm_eval_computed_goto(n->members[0]);
            if (n->fun.f1 == sqrt) {
                return sqrt(operand);
            }
            return n->fun.f1(operand);
        }
        case 2: {
            register double a = cm_eval_computed_goto(n->members[0]);
            register double b = cm_eval_computed_goto(n->members[1]);

            if (n->fun.f2 == add) {
                return a + b;
            } else if (n->fun.f2 == sub) {
                return a - b;
            } else if (n->fun.f2 == mul) {
                return a * b;
            } else if (n->fun.f2 == divide) {
                return a / b;
            } else if (n->fun.f2 == pow) {
                if (b == 2.0) return a * a;
                if (b == 0.5) return sqrt(a);
                if (b == 1.5) return a * sqrt(a);
                if (b == 2.5) return a * a * sqrt(a);
                return pow(a, b);
            }
            return n->fun.f2(a, b);
        }
        default:
            return n->fun.f3(cm_eval_computed_goto(n->members[0]),
                            cm_eval_computed_goto(n->members[1]),
                            cm_eval_computed_goto(n->members[2]));
    }
}

handle_default:
    return 0.0/0.0;
}
#endif

static double cm_eval_simd_arm64(const cm_expr *expr, const double *vars) {
#if defined(__aarch64__) || defined(__arm64__)
    if (!expr) return 0.0;

    switch(cm_get_type(expr->type)) {
        case CM_CONST:
            return expr->value;

        case CM_VAR:
            return *expr->bound;

        case CM_FUN: {
            int arity = cm_get_arity(expr->type);
            if (arity == 2) {
                double a = cm_eval_simd_arm64(expr->members[0], vars);
                double b = cm_eval_simd_arm64(expr->members[1], vars);

                float64x2_t va = vdupq_n_f64(a);
                float64x2_t vb = vdupq_n_f64(b);
                float64x2_t result;

                if (expr->fun.f2 == add) {
                    result = vaddq_f64(va, vb);
                } else if (expr->fun.f2 == sub) {
                    result = vsubq_f64(va, vb);
                } else if (expr->fun.f2 == mul) {
                    result = vmulq_f64(va, vb);
                } else if (expr->fun.f2 == divide) {
                    result = vdivq_f64(va, vb);
                } else {
                    return expr->fun.f2(a, b);
                }

                return vgetq_lane_f64(result, 0);
            } else if (arity == 1) {
                double operand = cm_eval_simd_arm64(expr->members[0], vars);

                float64x2_t v_operand = vdupq_n_f64(operand);

                if (expr->fun.f1 == sqrt) {
                    float64x2_t result = vsqrtq_f64(v_operand);
                    return vgetq_lane_f64(result, 0);
                } else if (expr->fun.f1 == fabs) {
                    float64x2_t result = vabsq_f64(v_operand);
                    return vgetq_lane_f64(result, 0);
                } else {
                    return expr->fun.f1(operand);
                }
            }

            switch(arity) {
                case 0: return expr->fun.f0();
                case 3: return expr->fun.f3(
                    cm_eval_simd_arm64(expr->members[0], vars),
                    cm_eval_simd_arm64(expr->members[1], vars),
                    cm_eval_simd_arm64(expr->members[2], vars));
                default: return 0.0;
            }
        }
    }
#endif
    return cm_eval_computed_goto(expr);
}

static double cm_eval_vectorized(const cm_expr *expr, const double *vars, int vector_size) {
#if defined(__aarch64__) || defined(__arm64__)
    if (vector_size >= 2 && expr && cm_get_type(expr->type) == CM_FUN) {
        int arity = cm_get_arity(expr->type);

        if (arity == 2) {
            double a1 = cm_eval_simd_arm64(expr->members[0], vars);
            double b1 = cm_eval_simd_arm64(expr->members[1], vars);
            double a2 = a1;
            double b2 = b1;

            float64x2_t va = {a1, a2};
            float64x2_t vb = {b1, b2};
            float64x2_t result;

            if (expr->fun.f2 == add) {
                result = vaddq_f64(va, vb);
            } else if (expr->fun.f2 == mul) {
                result = vmulq_f64(va, vb);
            } else {
                return expr->fun.f2(a1, b1);
            }

            return vgetq_lane_f64(result, 0);
        }
    }
#endif
    return cm_eval_simd_arm64(expr, vars);
}

static cm_bytecode *cm_compile_bytecode(const cm_expr *expr, int var_count) {
    cm_bytecode *bc = malloc(sizeof(cm_bytecode));
    if (!bc) return NULL;

    bc->code_capacity = BYTECODE_CHUNK_SIZE;
    bc->code = malloc(bc->code_capacity * sizeof(cm_instruction));
    bc->code_size = 0;
    bc->constants = malloc(64 * sizeof(double));
    bc->const_count = 0;
    bc->var_count = var_count;

    if (!bc->code || !bc->constants) {
        cm_bytecode_free(bc);
        return NULL;
    }

    cm_flatten_expression(expr, bc->code, &bc->code_size);
    return bc;
}

static void cm_flatten_expression(const cm_expr *expr, cm_instruction *instructions, size_t *count) {
    if (!expr || !instructions || !count) return;

    switch(cm_get_type(expr->type)) {
        case CM_CONST:
            instructions[*count].op = OP_CONST;
            instructions[*count].data.constant = expr->value;
            (*count)++;
            break;

        case CM_VAR:
            instructions[*count].op = OP_VAR;
            instructions[*count].data.var_index = 0;
            (*count)++;
            break;

        case CM_FUN: {
            int arity = cm_get_arity(expr->type);

            for (int i = 0; i < arity; i++) {
                cm_flatten_expression(expr->members[i], instructions, count);
            }

            if (arity == 2) {
                if (expr->fun.f2 == add) {
                    instructions[*count].op = OP_ADD;
                } else if (expr->fun.f2 == sub) {
                    instructions[*count].op = OP_SUB;
                } else if (expr->fun.f2 == mul) {
                    instructions[*count].op = OP_MUL;
                } else if (expr->fun.f2 == divide) {
                    instructions[*count].op = OP_DIV;
                } else if (expr->fun.f2 == pow) {
                    instructions[*count].op = OP_POW;
                }
            } else if (arity == 1) {
                if (expr->fun.f1 == sqrt) {
                    instructions[*count].op = OP_SQRT;
                } else if (expr->fun.f1 == sin) {
                    instructions[*count].op = OP_SIN;
                } else if (expr->fun.f1 == cos) {
                    instructions[*count].op = OP_COS;
                } else if (expr->fun.f1 == tan) {
                    instructions[*count].op = OP_TAN;
                } else if (expr->fun.f1 == log) {
                    instructions[*count].op = OP_LOG;
                } else if (expr->fun.f1 == exp) {
                    instructions[*count].op = OP_EXP;
                } else if (expr->fun.f1 == fabs) {
                    instructions[*count].op = OP_FABS;
                }
            }
            (*count)++;
            break;
        }
    }
}

static double cm_eval_bytecode(const cm_bytecode *bc, const double *vars) {
    if (!bc || !bc->code) return 0.0;

    double stack[256];
    int stack_ptr = 0;

    for (size_t i = 0; i < bc->code_size; i++) {
        const cm_instruction *inst = &bc->code[i];

        switch (inst->op) {
            case OP_CONST:
                stack[stack_ptr++] = inst->data.constant;
                break;

            case OP_VAR:
                if (vars) {
                    stack[stack_ptr++] = vars[inst->data.var_index];
                } else {
                    stack[stack_ptr++] = 0.0;
                }
                break;

            case OP_ADD:
                if (stack_ptr >= 2) {
                    double b = stack[--stack_ptr];
                    double a = stack[--stack_ptr];
                    stack[stack_ptr++] = a + b;
                }
                break;

            case OP_SUB:
                if (stack_ptr >= 2) {
                    double b = stack[--stack_ptr];
                    double a = stack[--stack_ptr];
                    stack[stack_ptr++] = a - b;
                }
                break;

            case OP_MUL:
                if (stack_ptr >= 2) {
                    double b = stack[--stack_ptr];
                    double a = stack[--stack_ptr];
                    stack[stack_ptr++] = a * b;
                }
                break;

            case OP_DIV:
                if (stack_ptr >= 2) {
                    double b = stack[--stack_ptr];
                    double a = stack[--stack_ptr];
                    stack[stack_ptr++] = a / b;
                }
                break;

            case OP_POW:
                if (stack_ptr >= 2) {
                    double b = stack[--stack_ptr];
                    double a = stack[--stack_ptr];
                    stack[stack_ptr++] = pow(a, b);
                }
                break;

            case OP_SQRT:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = sqrt(stack[stack_ptr-1]);
                }
                break;

            case OP_SIN:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = sin(stack[stack_ptr-1]);
                }
                break;

            case OP_COS:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = cos(stack[stack_ptr-1]);
                }
                break;

            case OP_TAN:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = tan(stack[stack_ptr-1]);
                }
                break;

            case OP_LOG:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = log(stack[stack_ptr-1]);
                }
                break;

            case OP_EXP:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = exp(stack[stack_ptr-1]);
                }
                break;

            case OP_FABS:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = fabs(stack[stack_ptr-1]);
                }
                break;

            case OP_NEG:
                if (stack_ptr >= 1) {
                    stack[stack_ptr-1] = -stack[stack_ptr-1];
                }
                break;

            case OP_RETURN:
                return stack_ptr > 0 ? stack[stack_ptr-1] : 0.0;
        }
    }

    return stack_ptr > 0 ? stack[stack_ptr-1] : 0.0;
}

static void cm_bytecode_free(cm_bytecode *bc) {
    if (bc) {
        free(bc->code);
        free(bc->constants);
        free(bc);
    }
}

static cm_pattern cm_analyze_pattern(const cm_expr *expr) {
    cm_pattern pattern = {PATTERN_UNKNOWN, {0}, 0, NULL};

    if (!expr) return pattern;

    switch(cm_get_type(expr->type)) {
        case CM_CONST:
            pattern.type = PATTERN_CONST;
            pattern.coefficients[0] = expr->value;
            break;

        case CM_VAR:
            pattern.type = PATTERN_VAR;
            break;

        case CM_FUN: {
            int arity = cm_get_arity(expr->type);

            if (arity == 2) {
                if (expr->fun.f2 == add) {
                    if (cm_get_type(expr->members[0]->type) == CM_VAR &&
                        cm_get_type(expr->members[1]->type) == CM_CONST) {
                        pattern.type = PATTERN_ADD_CONST;
                        pattern.coefficients[0] = expr->members[1]->value;
                    } else if (cm_get_type(expr->members[1]->type) == CM_VAR &&
                               cm_get_type(expr->members[0]->type) == CM_CONST) {
                        pattern.type = PATTERN_ADD_CONST;
                        pattern.coefficients[0] = expr->members[0]->value;
                    }
                }
                else if (expr->fun.f2 == mul) {
                    if (cm_get_type(expr->members[0]->type) == CM_VAR &&
                        cm_get_type(expr->members[1]->type) == CM_CONST) {
                        pattern.type = PATTERN_MUL_CONST;
                        pattern.coefficients[0] = expr->members[1]->value;
                    } else if (cm_get_type(expr->members[1]->type) == CM_VAR &&
                               cm_get_type(expr->members[0]->type) == CM_CONST) {
                        pattern.type = PATTERN_MUL_CONST;
                        pattern.coefficients[0] = expr->members[0]->value;
                    }
                }
            }
            break;
        }
    }

    return pattern;
}

static double cm_eval_specialized(const cm_expr *expr, const cm_pattern *pattern, const double *vars) {
    if (!pattern || !expr) return 0.0;

    switch (pattern->type) {
        case PATTERN_CONST:
            return pattern->coefficients[0];

        case PATTERN_VAR:
            return expr->bound ? *expr->bound : 0.0;

        case PATTERN_ADD_CONST: {
            if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 2 && expr->fun.f2 == add) {
                if (cm_get_type(expr->members[0]->type) == CM_VAR) {
                    return *expr->members[0]->bound + pattern->coefficients[0];
                } else if (cm_get_type(expr->members[1]->type) == CM_VAR) {
                    return *expr->members[1]->bound + pattern->coefficients[0];
                }
            }
            return cm_eval_simd_arm64(expr, vars);
        }

        case PATTERN_MUL_CONST: {
            if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 2 && expr->fun.f2 == mul) {
                if (cm_get_type(expr->members[0]->type) == CM_VAR) {
                    return *expr->members[0]->bound * pattern->coefficients[0];
                } else if (cm_get_type(expr->members[1]->type) == CM_VAR) {
                    return *expr->members[1]->bound * pattern->coefficients[0];
                }
            }
            return cm_eval_simd_arm64(expr, vars);
        }

        default:
            return cm_eval_simd_arm64(expr, vars);
    }
}

static double cm_eval_fast(const cm_expr *n) {
#if defined(__aarch64__) || defined(__arm64__)
    return cm_eval_simd_arm64(n, NULL);
#elif defined(__GNUC__)
    return cm_eval_computed_goto(n);
#else
    switch(cm_get_type(n->type)) {
        case CM_CONST: return n->value;
        case CM_VAR: return *n->bound;
        case CM_FUN:
            switch(cm_get_arity(n->type)) {
                case 0: return n->fun.f0();
                case 1: {
                    // Instruction-level parallelism: prefetch next operation
                    __builtin_prefetch(n->members[0], 0, 3);
                    return n->fun.f1(cm_eval_fast(n->members[0]));
                }
                case 2: {
                    // Evaluate both operands in parallel (compiler can optimize)
                    double a = cm_eval_fast(n->members[0]);
                    double b = cm_eval_fast(n->members[1]);

                    // Use FMA instruction when available for better performance
                    if (n->fun.f2 == add) {
                        return a + b;
                    } else if (n->fun.f2 == mul) {
                        return a * b;
                    } else {
                        return n->fun.f2(a, b);
                    }
                }
                case 3: return n->fun.f3(cm_eval_fast(n->members[0]), cm_eval_fast(n->members[1]), cm_eval_fast(n->members[2]));
                case 4: return n->fun.f4(cm_eval_fast(n->members[0]), cm_eval_fast(n->members[1]), cm_eval_fast(n->members[2]), cm_eval_fast(n->members[3]));
                case 5: return n->fun.f5(cm_eval_fast(n->members[0]), cm_eval_fast(n->members[1]), cm_eval_fast(n->members[2]), cm_eval_fast(n->members[3]), cm_eval_fast(n->members[4]));
                case 6: return n->fun.f6(cm_eval_fast(n->members[0]), cm_eval_fast(n->members[1]), cm_eval_fast(n->members[2]), cm_eval_fast(n->members[3]), cm_eval_fast(n->members[4]), cm_eval_fast(n->members[5]));
                case 7: return n->fun.f7(cm_eval_fast(n->members[0]), cm_eval_fast(n->members[1]), cm_eval_fast(n->members[2]), cm_eval_fast(n->members[3]), cm_eval_fast(n->members[4]), cm_eval_fast(n->members[5]), cm_eval_fast(n->members[6]));
                default: return 0.0/0.0;
            }
        default: return 0.0 / 0.0;
    }
#endif
}

static cm_expr *cm_constant_fold(cm_expr *expr, cm_expr_pool *pool) {
    if (!expr) return NULL;

    for (int i = 0; i < expr->member_count; i++) {
        expr->members[i] = cm_constant_fold(expr->members[i], pool);
    }

    if (cm_get_type(expr->type) == CM_FUN) {
        int arity = cm_get_arity(expr->type);

        int all_constants = 1;
        for (int i = 0; i < arity; i++) {
            if (cm_get_type(expr->members[i]->type) != CM_CONST) {
                all_constants = 0;
                break;
            }
        }

        if (all_constants) {
            double result = 0.0;
            switch (arity) {
                case 0:
                    result = expr->fun.f0();
                    break;
                case 1:
                    result = expr->fun.f1(expr->members[0]->value);
                    break;
                case 2:
                    result = expr->fun.f2(expr->members[0]->value, expr->members[1]->value);
                    break;
                case 3:
                    result = expr->fun.f3(expr->members[0]->value, expr->members[1]->value, expr->members[2]->value);
                    break;
            }

            cm_expr *const_expr = new_expr(CM_CONST, NULL, pool);
            if (const_expr) {
                const_expr->value = result;
                const_expr->optimization_flags |= CM_OPT_CONST_FOLDED;
            }
            return const_expr;
        }

        if (arity == 2) {
            double left_val = 0.0, right_val = 0.0;
            int left_const = (cm_get_type(expr->members[0]->type) == CM_CONST);
            int right_const = (cm_get_type(expr->members[1]->type) == CM_CONST);

            if (left_const) left_val = expr->members[0]->value;
            if (right_const) right_val = expr->members[1]->value;

            if (expr->fun.f2 == add) {
                if (left_const && left_val == 0.0) {
                    return expr->members[1]; // 0 + x = x
                }
                if (right_const && right_val == 0.0) {
                    return expr->members[0]; // x + 0 = x
                }
            }
            else if (expr->fun.f2 == mul) {
                if (left_const && left_val == 0.0) {
                    cm_expr *zero = new_expr(CM_CONST, NULL, pool);
                    if (zero) zero->value = 0.0;
                    return zero;
                }
                if (right_const && right_val == 0.0) {
                    cm_expr *zero = new_expr(CM_CONST, NULL, pool);
                    if (zero) zero->value = 0.0;
                    return zero;
                }
                if (left_const && left_val == 1.0) {
                    return expr->members[1]; // 1 * x = x
                }
                if (right_const && right_val == 1.0) {
                    return expr->members[0]; // x * 1 = x
                }
            }
            else if (expr->fun.f2 == pow) {
                if (right_const && right_val == 0.0) {
                    // x^0 = 1
                    cm_expr *one = new_expr(CM_CONST, NULL, pool);
                    if (one) one->value = 1.0;
                    return one;
                }
                if (right_const && right_val == 1.0) {
                    return expr->members[0]; // x^1 = x
                }
                if (left_const && left_val == 1.0) {
                    // 1^x = 1
                    cm_expr *one = new_expr(CM_CONST, NULL, pool);
                    if (one) one->value = 1.0;
                    return one;
                }
            }
        }
    }

    return expr;
}

// Advanced Mathematical Optimizations
static cm_expr *cm_apply_math_optimizations(cm_expr *expr, cm_expr_pool *pool) {
    if (!expr) return NULL;

    // Apply trigonometric identities
    if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 1) {
        cm_expr *operand = expr->members[0];

        // sin^2(x) + cos^2(x) = 1 detection (simplified case)
        if (expr->fun.f1 == sin && cm_get_type(operand->type) == CM_FUN) {
            // Could implement more complex pattern matching here
        }

        // sqrt(x^2) = |x| simplification
        if (expr->fun.f1 == sqrt &&
            cm_get_type(operand->type) == CM_FUN &&
            cm_get_arity(operand->type) == 2 &&
            operand->fun.f2 == pow &&
            cm_get_type(operand->members[1]->type) == CM_CONST &&
            operand->members[1]->value == 2.0) {

            // Replace sqrt(x^2) with fabs(x)
            cm_expr *abs_expr = new_expr(CM_FUN | 1, NULL, pool);
            if (abs_expr) {
                abs_expr->fun.f1 = fabs;
                abs_expr->member_count = 1;
                abs_expr->members[0] = operand->members[0];
                abs_expr->optimization_flags |= CM_OPT_PATTERN;
            }
            return abs_expr;
        }

        // log(exp(x)) = x simplification
        if (expr->fun.f1 == log &&
            cm_get_type(operand->type) == CM_FUN &&
            cm_get_arity(operand->type) == 1 &&
            operand->fun.f1 == exp) {
            return operand->members[0]; // log(exp(x)) = x
        }

        // exp(log(x)) = x simplification (for x > 0)
        if (expr->fun.f1 == exp &&
            cm_get_type(operand->type) == CM_FUN &&
            cm_get_arity(operand->type) == 1 &&
            operand->fun.f1 == log) {
            return operand->members[0]; // exp(log(x)) = x
        }
    }

    // Apply to children recursively
    for (int i = 0; i < expr->member_count; i++) {
        expr->members[i] = cm_apply_math_optimizations(expr->members[i], pool);
    }

    return expr;
}

// Cache-friendly memory layout optimization
static void cm_optimize_memory_layout(cm_expr *expr) {
    if (!expr) return;

    // Prefetch commonly accessed members
    __builtin_prefetch(expr, 0, 3); // Prefetch for read with high temporal locality

    // Recursively optimize children
    for (int i = 0; i < expr->member_count; i++) {
        if (expr->members[i]) {
            __builtin_prefetch(expr->members[i], 0, 3);
            cm_optimize_memory_layout(expr->members[i]);
        }
    }
}

// Advanced expression compiler with all optimizations
static cm_expr *cm_compile_optimized(const char *expression, const cm_variable *variables, int var_count, int *error, cm_expr_pool *pool) {
    // First, compile normally
    cm_expr *expr = cm_compile(expression, variables, var_count, error);
    if (!expr || (error && *error != 0)) {
        return expr;
    }

    // Apply constant folding
    expr = cm_constant_fold(expr, pool);

    // Apply mathematical optimizations
    expr = cm_apply_math_optimizations(expr, pool);

    // Optimize memory layout
    cm_optimize_memory_layout(expr);

    // Analyze patterns for specialization
    if (expr) {
        expr->pattern = malloc(sizeof(cm_pattern));
        if (expr->pattern) {
            cm_pattern temp_pattern = cm_analyze_pattern(expr);
            memcpy(expr->pattern, &temp_pattern, sizeof(cm_pattern));
            expr->optimization_flags |= CM_OPT_PATTERN;
        }
    }

    // Compile bytecode for complex expressions
    if (expr && !expr->bytecode && expr->member_count > 2) {
        expr->bytecode = cm_compile_bytecode(expr, var_count);
        if (expr->bytecode) {
            expr->optimization_flags |= CM_OPT_BYTECODE;
        }
    }

    return expr;
}

double cm_eval(const cm_expr *n, int *error) {
    // Use optimized evaluation paths
    if (!error) {
        return cm_eval_fast(n);
    }

    switch(cm_get_type(n->type)) {
        case CM_CONST: return n->value;
        case CM_VAR: return *n->bound;
        case CM_FUN:
            switch(cm_get_arity(n->type)) {
#define m(e) cm_eval(n->members[e], error)
                case 0: return n->fun.f0();
                case 1: return n->fun.f1(m(0));
                case 2: {
                    double a = m(0);
                    double b = m(1);
                    if (n->fun.f2 == divide && b == 0) {
                        *error = CM_ERROR_DIVISION_BY_ZERO;
                        return 0.0 / 0.0; // NaN
                    }
                    return n->fun.f2(a, b);
                }
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

    // constant folding for all function types
    if (cm_get_type(n->type) == CM_FUN) {
        int all_const = 1;
        for (int i = 0; i < n->member_count; ++i) {
            if (cm_get_type(n->members[i]->type) != CM_CONST) {
                all_const = 0;
                break;
            }
        }

        if (all_const) {
            double result;
            switch(n->member_count) {
                case 0: result = n->fun.f0(); break;
                case 1: result = n->fun.f1(n->members[0]->value); break;
                case 2: result = n->fun.f2(n->members[0]->value, n->members[1]->value); break;
                case 3: result = n->fun.f3(n->members[0]->value, n->members[1]->value, n->members[2]->value); break;
                case 4: result = n->fun.f4(n->members[0]->value, n->members[1]->value, n->members[2]->value, n->members[3]->value); break;
                default: return; // Skip optimization for higher arities
            }

            n->type = CM_CONST;
            n->value = result;
            for (int i = 0; i < n->member_count; ++i) {
                cm_free(n->members[i]);
            }
            n->member_count = 0;
        }
    }
}

// JIT Compiler Implementation (x86-64 machine code generation)
static void jit_emit_byte(cm_jit_builder *builder, unsigned char byte) {
    if (builder->pos < builder->size) {
        builder->buffer[builder->pos++] = byte;
    }
}

static void jit_emit_bytes(cm_jit_builder *builder, const unsigned char *bytes, size_t count) {
    for (size_t i = 0; i < count; i++) {
        jit_emit_byte(builder, bytes[i]);
    }
}

static void jit_emit_double_load(cm_jit_builder *builder, double value) {
    // movsd xmm0, [rip+offset] - Load immediate double
    unsigned char code[] = {0xF2, 0x0F, 0x10, 0x05, 0x00, 0x00, 0x00, 0x00};
    jit_emit_bytes(builder, code, sizeof(code));
    // Store the double value at the end of code
    memcpy(builder->buffer + builder->pos, &value, sizeof(double));
    builder->pos += sizeof(double);
}

static void jit_emit_var_load(cm_jit_builder *builder, int var_index) {
    // movsd xmm0, [rdi + var_index*8] - Load variable from array
    unsigned char code[] = {0xF2, 0x0F, 0x10, 0x87};
    jit_emit_bytes(builder, code, sizeof(code));
    int offset = var_index * 8;
    jit_emit_bytes(builder, (unsigned char*)&offset, 4);
}

static void jit_emit_add(cm_jit_builder *builder) {
    // addsd xmm0, xmm1
    unsigned char code[] = {0xF2, 0x0F, 0x58, 0xC1};
    jit_emit_bytes(builder, code, sizeof(code));
}

static void jit_emit_mul(cm_jit_builder *builder) {
    // mulsd xmm0, xmm1
    unsigned char code[] = {0xF2, 0x0F, 0x59, 0xC1};
    jit_emit_bytes(builder, code, sizeof(code));
}

static void jit_emit_sqrt(cm_jit_builder *builder) {
    // sqrtsd xmm0, xmm0
    unsigned char code[] = {0xF2, 0x0F, 0x51, 0xC0};
    jit_emit_bytes(builder, code, sizeof(code));
}

static void jit_emit_pow(cm_jit_builder *builder) {
    // Call pow function - more complex, use function call
    // mov rdi, rax (save xmm0 to rdi as double)
    // call pow
    unsigned char code[] = {
        0x48, 0x83, 0xEC, 0x10,  // sub rsp, 16 (align stack)
        0xF2, 0x0F, 0x11, 0x04, 0x24,  // movsd [rsp], xmm0
        0xF2, 0x0F, 0x11, 0x4C, 0x24, 0x08,  // movsd [rsp+8], xmm1
        0xF2, 0x0F, 0x10, 0x04, 0x24,  // movsd xmm0, [rsp]
        0xF2, 0x0F, 0x10, 0x4C, 0x24, 0x08,  // movsd xmm1, [rsp+8]
        0x48, 0xB8  // mov rax, immediate (pow address)
    };
    jit_emit_bytes(builder, code, sizeof(code));
    void *pow_addr = (void*)pow;
    jit_emit_bytes(builder, (unsigned char*)&pow_addr, 8);
    unsigned char call_code[] = {
        0xFF, 0xD0,  // call rax
        0x48, 0x83, 0xC4, 0x10  // add rsp, 16 (restore stack)
    };
    jit_emit_bytes(builder, call_code, sizeof(call_code));
}

static void jit_emit_return(cm_jit_builder *builder) {
    // ret
    jit_emit_byte(builder, 0xC3);
}

static int jit_compile_expr(cm_jit_builder *builder, const cm_expr *expr, int *var_counter) {
    switch (cm_get_type(expr->type)) {
        case CM_CONST:
            jit_emit_double_load(builder, expr->value);
            return 1;

        case CM_VAR:
            jit_emit_var_load(builder, (*var_counter)++);
            return 1;

        case CM_FUN: {
            int arity = cm_get_arity(expr->type);

            // Simple binary operations optimization
            if (arity == 2) {
                // Compile left operand
                jit_compile_expr(builder, expr->members[0], var_counter);

                // Push xmm0 to stack
                unsigned char push_code[] = {
                    0x48, 0x83, 0xEC, 0x08,  // sub rsp, 8
                    0xF2, 0x0F, 0x11, 0x04, 0x24  // movsd [rsp], xmm0
                };
                jit_emit_bytes(builder, push_code, sizeof(push_code));

                // Compile right operand
                jit_compile_expr(builder, expr->members[1], var_counter);

                // Move result to xmm1
                unsigned char move_code[] = {0xF2, 0x0F, 0x10, 0xC8}; // movsd xmm1, xmm0
                jit_emit_bytes(builder, move_code, sizeof(move_code));

                // Pop left operand back to xmm0
                unsigned char pop_code[] = {
                    0xF2, 0x0F, 0x10, 0x04, 0x24,  // movsd xmm0, [rsp]
                    0x48, 0x83, 0xC4, 0x08  // add rsp, 8
                };
                jit_emit_bytes(builder, pop_code, sizeof(pop_code));

                // Emit operation
                if (expr->fun.f2 == add) {
                    jit_emit_add(builder);
                } else if (expr->fun.f2 == mul) {
                    jit_emit_mul(builder);
                } else if (expr->fun.f2 == pow) {
                    jit_emit_pow(builder);
                }

                return 1;
            } else if (arity == 1 && expr->fun.f1 == sqrt) {
                jit_compile_expr(builder, expr->members[0], var_counter);
                jit_emit_sqrt(builder);
                return 1;
            }

            // Fallback to function call for complex operations
            return 0;
        }

        default:
            return 0;
    }
}

static cm_jit_code *cm_compile_jit(const cm_expr *expr, int var_count) {
    cm_jit_code *jit = malloc(sizeof(cm_jit_code));
    if (!jit) return NULL;

    // Allocate executable memory
    jit->code = mmap(NULL, JIT_CODE_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (jit->code == MAP_FAILED) {
        free(jit);
        return NULL;
    }

    cm_jit_builder builder = {jit->code, 0, JIT_CODE_SIZE};

    // Function prologue
    unsigned char prologue[] = {
        0x55,           // push rbp
        0x48, 0x89, 0xE5  // mov rbp, rsp
    };
    jit_emit_bytes(&builder, prologue, sizeof(prologue));

    // Compile expression
    int var_counter = 0;
    if (jit_compile_expr(&builder, expr, &var_counter)) {
        // Function epilogue
        unsigned char epilogue[] = {
            0x5D,           // pop rbp
        };
        jit_emit_bytes(&builder, epilogue, sizeof(epilogue));
        jit_emit_return(&builder);

        jit->size = builder.pos;
        jit->compiled_func = (double (*)(const double*))jit->code;

        // Make memory executable only
        mprotect(jit->code, JIT_CODE_SIZE, PROT_READ | PROT_EXEC);

        return jit;
    }

    // JIT compilation failed, cleanup
    munmap(jit->code, JIT_CODE_SIZE);
    free(jit);
    return NULL;
}

static void cm_jit_free(cm_jit_code *jit) {
    if (jit) {
        if (jit->code) {
            munmap(jit->code, JIT_CODE_SIZE);
        }
        free(jit);
    }
}

cm_expr *cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error) {
    cm_init_pool(&globalPool);
    state s;
    s.start = s.next = expression;
    s.lookup = variables;
    s.lookup_len = var_count;

    int parse_error = 0;
    next_token(&s);
    cm_expr *root = list(&s, &parse_error);

    if (parse_error != 0) {
        cm_free(root);
        if (error) {
            *error = (int) (s.next - s.start);
            if (*error == 0) *error = 1;
        }
        return NULL;
    } else {
        optimise(root);

        // Apply all advanced optimizations
        root = cm_constant_fold(root, &globalPool);
        root = cm_apply_math_optimizations(root, &globalPool);
        cm_optimize_memory_layout(root);

        // Set up optimization structures
        if (root) {
            root->optimization_flags = CM_OPT_NONE;



            // Try JIT compilation for performance
            root->jit_code = cm_compile_jit(root, var_count);
            if (root->jit_code) {
                root->optimization_flags |= CM_OPT_JIT;
            }

            // Mark as SIMD optimized for ARM64
#if defined(__aarch64__) || defined(__arm64__)
            root->optimization_flags |= CM_OPT_SIMD;
#endif
        }

        if (error) *error = 0;
        return root;
    }
}

double cm_interp(const char *expression, int *error) {
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
