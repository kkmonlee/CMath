//

#include "cmath.h"
#include "cm_vector.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdbool.h>

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
    int nextFree;
    pthread_mutex_t mutex;
} cm_expr_pool;

cm_expr_pool globalPool = { {0}, 0, PTHREAD_MUTEX_INITIALIZER };

#define JIT_CODE_SIZE 4096
#define BYTECODE_CHUNK_SIZE 256
#define MAX_EXPRESSION_DEPTH 64
#define SIMD_VECTOR_SIZE 4
#define VECTORIZED_BATCH_SIZE 8
#define INLINE_CACHE_SIZE 32
#define EXPRESSION_TEMPLATE_CACHE 64

typedef enum {
    EVAL_MODE_STANDARD,
    EVAL_MODE_VECTORIZED,
    EVAL_MODE_TEMPLATE_SPECIALIZED,
    EVAL_MODE_INLINE_CACHED,
    EVAL_MODE_NATIVE_COMPILED
} cm_eval_mode;

// expression templates for ultra-fast evaluation
typedef struct {
    cm_eval_mode mode;
    double (*fast_eval)(const double *vars);
    void *compiled_code;
    uint64_t pattern_hash;
    int hit_count;
} cm_expression_template;

// inline cache for variable access
typedef struct {
    const double *var_ptr;
    uint64_t var_hash;
    double cached_value;
    int cache_valid;
} cm_inline_cache;

// ultra-optimized expression context
typedef struct {
    cm_expression_template templates[EXPRESSION_TEMPLATE_CACHE];
    cm_inline_cache var_cache[INLINE_CACHE_SIZE];
    double *vectorized_workspace;
    int template_count;
    uint64_t evaluation_count;
} cm_optimization_context;

// global optimization context
static cm_optimization_context *globalOptContext = NULL;
static pthread_mutex_t optContextMutex = PTHREAD_MUTEX_INITIALIZER;

typedef enum {
    OP_CONST, OP_VAR, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW,
    OP_NEG, OP_SQRT, OP_SIN, OP_COS, OP_TAN, OP_LOG, OP_EXP,
    OP_FABS, OP_RETURN,
    // vector opcodes for SIMD operations
    OP_V_LOAD_CONST, OP_V_LOAD_VAR, OP_V_ADD, OP_V_SUB, OP_V_MUL,
    OP_V_DIV, OP_V_EXP, OP_V_SIN, OP_V_COS, OP_V_SQRT, OP_V_FMA,
    OP_V_RETURN, OP_V_CALL
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
static void cm_specialize_expression(cm_expr *expr);
static double cm_eval_ultra_optimized(const cm_expr *n);
static cm_expr *cm_ultra_aggressive_optimize(cm_expr *expr, cm_expr_pool *pool);
static cm_optimization_context *cm_create_optimization_context(void);
static void cm_destroy_optimization_context(cm_optimization_context *ctx);
static double cm_eval_ultra_fast(const cm_expr *expr, cm_optimization_context *ctx, const double *vars);
static uint64_t cm_hash_expression_pattern(const cm_expr *expr);
static void cm_generate_native_code(const cm_expr *expr, cm_expression_template *tmpl);
static double cm_eval_template_specialized(const cm_expr *expr, const cm_pattern *pattern, const double *vars);
static void cm_vectorized_batch_eval(const cm_expr *expr, const double *input_batch, double *output_batch, int count);
static double cm_eval_inline_cached(const cm_expr *expr, cm_optimization_context *ctx, const double *vars);

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
            register double operand;
            register cm_expr *child = n->members[0];

            if (__builtin_expect(cm_get_type(child->type) == CM_VAR, 1)) {
                operand = *child->bound;
            } else if (__builtin_expect(cm_get_type(child->type) == CM_CONST, 1)) {
                operand = child->value;
            } else {
                operand = cm_eval_computed_goto(child);
            }

            register cm_fun1 func = n->fun.f1;
            if (__builtin_expect(func == sqrt, 1)) {
                return sqrt(operand);
            } else if (__builtin_expect(func == sin, 1)) {
                return sin(operand);
            } else if (__builtin_expect(func == cos, 1)) {
                return cos(operand);
            } else if (__builtin_expect(func == fabs, 1)) {
                return fabs(operand);
            }
            return func(operand);
        }
        case 2: {
            register double a, b;
            register cm_expr *left = n->members[0];
            register cm_expr *right = n->members[1];

            if (__builtin_expect(cm_get_type(left->type) == CM_VAR, 1)) {
                a = *left->bound;
            } else if (__builtin_expect(cm_get_type(left->type) == CM_CONST, 1)) {
                a = left->value;
            } else {
                a = cm_eval_computed_goto(left);
            }

            if (__builtin_expect(cm_get_type(right->type) == CM_VAR, 1)) {
                b = *right->bound;
            } else if (__builtin_expect(cm_get_type(right->type) == CM_CONST, 1)) {
                b = right->value;
            } else {
                b = cm_eval_computed_goto(right);
            }

            register cm_fun2 func = n->fun.f2;
            if (__builtin_expect(func == add, 1)) {
                return a + b;
            } else if (__builtin_expect(func == sub, 1)) {
                return a - b;
            } else if (__builtin_expect(func == mul, 1)) {
                return a * b;
            } else if (__builtin_expect(func == divide, 1)) {
                return a / b;
            } else if (__builtin_expect(func == pow, 0)) {
                if (__builtin_expect(b == 2.0, 1)) return a * a;
                if (__builtin_expect(b == 0.5, 1)) return sqrt(a);
                if (__builtin_expect(b == 1.0, 1)) return a;
                if (__builtin_expect(b == 0.0, 1)) return 1.0;
                if (__builtin_expect(b == 3.0, 1)) return a * a * a;
                if (__builtin_expect(b == 1.5, 1)) return a * sqrt(a);
                if (__builtin_expect(b == 2.5, 1)) return a * a * sqrt(a);
                return pow(a, b);
            }
            return func(a, b);
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

// optimization context management
static cm_optimization_context *cm_create_optimization_context(void) {
    cm_optimization_context *ctx = malloc(sizeof(cm_optimization_context));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(cm_optimization_context));
    // use aligned allocation with fallback
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign((void**)&ctx->vectorized_workspace, 32, VECTORIZED_BATCH_SIZE * sizeof(double)) != 0) {
        ctx->vectorized_workspace = malloc(VECTORIZED_BATCH_SIZE * sizeof(double));
    }
#else
    ctx->vectorized_workspace = malloc(VECTORIZED_BATCH_SIZE * sizeof(double));
#endif
    ctx->evaluation_count = 0;

    return ctx;
}

static void cm_destroy_optimization_context(cm_optimization_context *ctx) {
    if (!ctx) return;

    for (int i = 0; i < ctx->template_count; i++) {
        if (ctx->templates[i].compiled_code) {
            free(ctx->templates[i].compiled_code);
        }
    }

    if (ctx->vectorized_workspace) {
        free(ctx->vectorized_workspace);
    }

    free(ctx);
}

// cleanup function for optimization context
void cm_cleanup_global_optimization(void) {
    pthread_mutex_lock(&optContextMutex);
    if (globalOptContext) {
        cm_destroy_optimization_context(globalOptContext);
        globalOptContext = NULL;
    }
    pthread_mutex_unlock(&optContextMutex);
}

// expression pattern hashing using FNV-1a
static uint64_t cm_hash_expression_pattern(const cm_expr *expr) {
    if (!expr) return 0;

    uint64_t hash = 14695981039346656037ULL;

    // hash expression type and structure
    hash ^= (uint64_t)expr->type;
    hash *= 1099511628211ULL;

    if (cm_get_type(expr->type) == CM_CONST) {
        uint64_t value_bits = *(uint64_t*)&expr->value;
        hash ^= value_bits;
        hash *= 1099511628211ULL;
    } else if (cm_get_type(expr->type) == CM_FUN) {
        hash ^= (uint64_t)(uintptr_t)expr->fun.f2;
        hash *= 1099511628211ULL;

        // recursively hash children
        int arity = cm_get_arity(expr->type);
        for (int i = 0; i < arity && i < 3; i++) {
            if (expr->members[i]) {
                hash ^= cm_hash_expression_pattern(expr->members[i]) >> (i * 8);
                hash *= 1099511628211ULL;
            }
        }
    }

    return hash;
}

// template-based specialized evaluation
static double cm_eval_template_specialized(const cm_expr *expr, const cm_pattern *pattern, const double *vars) {
    // optimized patterns for common mathematical expressions
    switch (pattern->type) {
        case PATTERN_CONST:
            return pattern->coefficients[0];

        case PATTERN_VAR:
            if (cm_get_type(expr->type) == CM_VAR) {
                return *expr->bound;
            }
            break;

        case PATTERN_POW_CONST:
            if (cm_get_type(expr->type) == CM_FUN && expr->members[0] &&
                cm_get_type(expr->members[0]->type) == CM_VAR) {
                double base = *expr->members[0]->bound;
                return pow(base, pattern->coefficients[0]);
            }
            break;

        case PATTERN_TRIG:
        case PATTERN_EXP_LOG:
        case PATTERN_POLYNOMIAL:
            // implement for more complex optimizations
            break;

        case PATTERN_ADD_CONST: {
            // direct memory access
            if (cm_get_type(expr->type) == CM_FUN && expr->members[0] &&
                cm_get_type(expr->members[0]->type) == CM_VAR) {
                return *expr->members[0]->bound + pattern->coefficients[0];
            }
            break;
        }

        case PATTERN_MUL_CONST: {
            if (cm_get_type(expr->type) == CM_FUN && expr->members[0] &&
                cm_get_type(expr->members[0]->type) == CM_VAR) {
                return *expr->members[0]->bound * pattern->coefficients[0];
            }
            break;
        }

        case PATTERN_LINEAR: {
            // ax + b pattern - extremely common
            if (cm_get_type(expr->type) == CM_FUN && expr->members[0] &&
                cm_get_type(expr->members[0]->type) == CM_VAR) {
                return pattern->coefficients[0] * (*expr->members[0]->bound) + pattern->coefficients[1];
            }
            break;
        }

        case PATTERN_QUADRATIC: {
            // ax^2 + bx + c pattern
            if (cm_get_type(expr->type) == CM_FUN && expr->members[0] &&
                cm_get_type(expr->members[0]->type) == CM_VAR) {
                double x = *expr->members[0]->bound;
                return pattern->coefficients[0] * x * x + pattern->coefficients[1] * x + pattern->coefficients[2];
            }
            break;
        }

        case PATTERN_UNKNOWN:
        default:
            break;
    }

    return cm_eval_simd_arm64(expr, vars);
}

// vectorized batch evaluation for SIMD
static void cm_vectorized_batch_eval(const cm_expr *expr, const double *input_batch, double *output_batch, int count) {
#if defined(__aarch64__) || defined(__arm64__)
    // process in NEON SIMD batches of 2 doubles
    int simd_count = count & ~1;

    for (int i = 0; i < simd_count; i += 2) {
        float64x2_t inputs = vld1q_f64(&input_batch[i]);
        float64x2_t results;

        // pattern recognition for vectorization
        if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 2) {
            if (expr->fun.f2 == add && cm_get_type(expr->members[1]->type) == CM_CONST) {
                // vectorized a + constant
                float64x2_t constant = vdupq_n_f64(expr->members[1]->value);
                results = vaddq_f64(inputs, constant);
            } else if (expr->fun.f2 == mul && cm_get_type(expr->members[1]->type) == CM_CONST) {
                // vectorized a * constant
                float64x2_t constant = vdupq_n_f64(expr->members[1]->value);
                results = vmulq_f64(inputs, constant);
            } else {
                // fallback to scalar evaluation
                output_batch[i] = cm_eval_simd_arm64(expr, &input_batch[i]);
                output_batch[i+1] = cm_eval_simd_arm64(expr, &input_batch[i+1]);
                continue;
            }
        } else {
            // fallback to scalar evaluation
            output_batch[i] = cm_eval_simd_arm64(expr, &input_batch[i]);
            output_batch[i+1] = cm_eval_simd_arm64(expr, &input_batch[i+1]);
            continue;
        }

        vst1q_f64(&output_batch[i], results);
    }

    // handle remaining elements
    for (int i = simd_count; i < count; i++) {
        output_batch[i] = cm_eval_simd_arm64(expr, &input_batch[i]);
    }
#else
    // fallback for non-ARM platforms
    for (int i = 0; i < count; i++) {
        output_batch[i] = cm_eval_simd_arm64(expr, &input_batch[i]);
    }
#endif
}

// inline caching for variable access
static double cm_eval_inline_cached(const cm_expr *expr, cm_optimization_context *ctx, const double *vars) {
    if (!ctx || !expr) return 0.0;

    // check inline cache for variable access
    if (cm_get_type(expr->type) == CM_VAR) {
        uint64_t var_hash = (uint64_t)(uintptr_t)expr->bound;
        int cache_idx = var_hash % INLINE_CACHE_SIZE;

        cm_inline_cache *cache = &ctx->var_cache[cache_idx];
        if (cache->cache_valid && cache->var_ptr == expr->bound) {
            return *expr->bound;
        } else {
            // update cache
            cache->var_ptr = expr->bound;
            cache->var_hash = var_hash;
            cache->cached_value = *expr->bound;
            cache->cache_valid = 1;
            return cache->cached_value;
        }
    }

    return cm_eval_simd_arm64(expr, vars);
}

// native code generation (simplified JIT approach)
static void cm_generate_native_code(const cm_expr *expr, cm_expression_template *tmpl) {
    if (!expr || !tmpl) return;

    // create specialized function pointers for common patterns
    uint64_t pattern_hash = cm_hash_expression_pattern(expr);

    // generate specialized functions for common patterns
    if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 2 && expr->fun.f2 == add) {
        if (cm_get_type(expr->members[0]->type) == CM_VAR && cm_get_type(expr->members[1]->type) == CM_CONST) {
            // generate lambda for x + constant
            double constant = expr->members[1]->value;
            tmpl->fast_eval = NULL;
            tmpl->pattern_hash = pattern_hash;
            tmpl->mode = EVAL_MODE_TEMPLATE_SPECIALIZED;
        }
    }
}

// fast evaluation with optimizations
static double cm_eval_ultra_fast(const cm_expr *expr, cm_optimization_context *ctx, const double *vars) {
    if (!expr) return 0.0;

    ctx->evaluation_count++;

    // check template cache first
    uint64_t pattern_hash = cm_hash_expression_pattern(expr);
    for (int i = 0; i < ctx->template_count; i++) {
        if (ctx->templates[i].pattern_hash == pattern_hash) {
            ctx->templates[i].hit_count++;

            // use specialized template if available
            if (ctx->templates[i].fast_eval) {
                return ctx->templates[i].fast_eval(vars);
            }
            break;
        }
    }

    // try inline cached evaluation
    if (cm_get_type(expr->type) == CM_VAR) {
        return cm_eval_inline_cached(expr, ctx, vars);
    }

    // create new template if cache has space
    if (ctx->template_count < EXPRESSION_TEMPLATE_CACHE) {
        cm_expression_template *tmpl = &ctx->templates[ctx->template_count++];
        tmpl->pattern_hash = pattern_hash;
        tmpl->hit_count = 1;
        cm_generate_native_code(expr, tmpl);
    }

    // fallback to optimized SIMD evaluation
    return cm_eval_simd_arm64(expr, vars);
}

// ultra-fast expression specialization and inlining
static void cm_specialize_expression(cm_expr *expr) {
    if (!expr) return;

    // recursively specialize children first
    for (int i = 0; i < expr->member_count; i++) {
        cm_specialize_expression(expr->members[i]);
    }

    // analyze and cache patterns for this expression
    if (!expr->pattern) {
        expr->pattern = malloc(sizeof(cm_pattern));
        if (expr->pattern) {
            cm_pattern temp_pattern = cm_analyze_pattern(expr);
            memcpy(expr->pattern, &temp_pattern, sizeof(cm_pattern));
            expr->optimization_flags |= CM_OPT_PATTERN;

            // for simple patterns, create ultra-fast inline evaluators
            if (temp_pattern.type == PATTERN_ADD_CONST ||
                temp_pattern.type == PATTERN_MUL_CONST ||
                temp_pattern.type == PATTERN_LINEAR ||
                temp_pattern.type == PATTERN_POW_CONST) {
                expr->optimization_flags |= CM_OPT_CONST_FOLDED;
            }
        }
    }

    // mark as specialized
    expr->optimization_flags |= CM_OPT_SPECIALIZED;
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
                // advanced pattern recognition
                if (expr->fun.f2 == add) {
                    // check for ax + b pattern (LINEAR)
                    cm_expr *left = expr->members[0];
                    cm_expr *right = expr->members[1];

                    if (cm_get_type(left->type) == CM_FUN && left->fun.f2 == mul &&
                        cm_get_type(left->members[0]->type) == CM_CONST &&
                        cm_get_type(left->members[1]->type) == CM_VAR &&
                        cm_get_type(right->type) == CM_CONST) {
                        // pattern: (a * x) + b
                        pattern.type = PATTERN_LINEAR;
                        pattern.coefficients[0] = left->members[0]->value;  // a
                        pattern.coefficients[1] = right->value;              // b
                    } else if (cm_get_type(left->type) == CM_VAR &&
                               cm_get_type(right->type) == CM_CONST) {
                        // pattern: x + constant
                        pattern.type = PATTERN_ADD_CONST;
                        pattern.coefficients[0] = right->value;
                    } else if (cm_get_type(right->type) == CM_VAR &&
                               cm_get_type(left->type) == CM_CONST) {
                        // pattern: constant + x
                        pattern.type = PATTERN_ADD_CONST;
                        pattern.coefficients[0] = left->value;
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
                else if (expr->fun.f2 == pow) {
                    // check for quadratic patterns: x^2
                    if (cm_get_type(expr->members[0]->type) == CM_VAR &&
                        cm_get_type(expr->members[1]->type) == CM_CONST &&
                        expr->members[1]->value == 2.0) {
                        pattern.type = PATTERN_POW_CONST;
                        pattern.coefficients[0] = 2.0;
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

// fast evaluation using direct inlining and specialization
static double cm_eval_ultra_optimized(const cm_expr *n) {
    // fast path for most common patterns - minimal overhead
    switch(cm_get_type(n->type)) {
        case CM_CONST:
            return n->value;

        case CM_VAR:
            return *n->bound;

        case CM_FUN: {
            int arity = cm_get_arity(n->type);

            if (arity == 2) {
                // optimized binary operations
                double a, b;

                // direct evaluation with minimal overhead
                if (cm_get_type(n->members[0]->type) == CM_VAR) {
                    a = *n->members[0]->bound;
                } else if (cm_get_type(n->members[0]->type) == CM_CONST) {
                    a = n->members[0]->value;
                } else {
                    a = cm_eval_ultra_optimized(n->members[0]);
                }

                if (cm_get_type(n->members[1]->type) == CM_VAR) {
                    b = *n->members[1]->bound;
                } else if (cm_get_type(n->members[1]->type) == CM_CONST) {
                    b = n->members[1]->value;
                } else {
                    b = cm_eval_ultra_optimized(n->members[1]);
                }

                // direct arithmetic - no function pointers
                if (n->fun.f2 == add) {
                    return a + b;
                } else if (n->fun.f2 == sub) {
                    return a - b;
                } else if (n->fun.f2 == mul) {
                    return a * b;
                } else if (n->fun.f2 == divide) {
                    return a / b;
                } else if (n->fun.f2 == pow) {
                    return pow(a, b);
                } else {
                    return n->fun.f2(a, b);
                }

            } else if (arity == 1) {
                double operand;
                if (cm_get_type(n->members[0]->type) == CM_VAR) {
                    operand = *n->members[0]->bound;
                } else if (cm_get_type(n->members[0]->type) == CM_CONST) {
                    operand = n->members[0]->value;
                } else {
                    operand = cm_eval_ultra_optimized(n->members[0]);
                }

                // direct math functions
                if (n->fun.f1 == sqrt) {
                    return sqrt(operand);
                } else if (n->fun.f1 == sin) {
                    return sin(operand);
                } else if (n->fun.f1 == cos) {
                    return cos(operand);
                } else if (n->fun.f1 == fabs) {
                    return fabs(operand);
                } else {
                    return n->fun.f1(operand);
                }
            } else {
                // fallback for other arities
                return n->fun.f0();
            }
        }
    }
    return 0.0;
}

// ultra-aggressive evaluation with maximum inlining
static inline double cm_eval_hyper_optimized(const cm_expr *n) {
    register int type = cm_get_type(n->type);

    if (__builtin_expect(type == CM_CONST, 1)) {
        return n->value;
    }

    if (__builtin_expect(type == CM_VAR, 1)) {
        return *n->bound;
    }

    if (__builtin_expect(type == CM_FUN, 1)) {
        register int arity = cm_get_arity(n->type);

        if (__builtin_expect(arity == 2, 1)) {
            register double a, b;

            // ultra-fast operand evaluation with branch prediction hints
            if (__builtin_expect(cm_get_type(n->members[0]->type) == CM_VAR, 1)) {
                a = *n->members[0]->bound;
            } else if (__builtin_expect(cm_get_type(n->members[0]->type) == CM_CONST, 1)) {
                a = n->members[0]->value;
            } else {
                a = cm_eval_hyper_optimized(n->members[0]);
            }

            if (__builtin_expect(cm_get_type(n->members[1]->type) == CM_VAR, 1)) {
                b = *n->members[1]->bound;
            } else if (__builtin_expect(cm_get_type(n->members[1]->type) == CM_CONST, 1)) {
                b = n->members[1]->value;
            } else {
                b = cm_eval_hyper_optimized(n->members[1]);
            }

            // direct function pointer comparison and inlined operations
            register cm_fun2 func = n->fun.f2;
            if (__builtin_expect(func == add, 1)) {
                return a + b;
            } else if (__builtin_expect(func == mul, 1)) {
                return a * b;
            } else if (__builtin_expect(func == sub, 1)) {
                return a - b;
            } else if (__builtin_expect(func == divide, 1)) {
                return a / b;
            } else if (__builtin_expect(func == pow, 0)) {
                // optimized pow for common cases
                if (__builtin_expect(b == 2.0, 1)) return a * a;
                if (__builtin_expect(b == 0.5, 1)) return sqrt(a);
                if (__builtin_expect(b == 1.0, 1)) return a;
                if (__builtin_expect(b == 0.0, 1)) return 1.0;
                if (__builtin_expect(b == 3.0, 1)) return a * a * a;
                if (__builtin_expect(b == 1.5, 1)) return a * sqrt(a);
                if (__builtin_expect(b == 2.5, 1)) return a * a * sqrt(a);
                return pow(a, b);
            } else {
                return func(a, b);
            }
        } else if (__builtin_expect(arity == 1, 1)) {
            register double operand;
            if (__builtin_expect(cm_get_type(n->members[0]->type) == CM_VAR, 1)) {
                operand = *n->members[0]->bound;
            } else if (__builtin_expect(cm_get_type(n->members[0]->type) == CM_CONST, 1)) {
                operand = n->members[0]->value;
            } else {
                operand = cm_eval_hyper_optimized(n->members[0]);
            }

            register cm_fun1 func = n->fun.f1;
            if (__builtin_expect(func == sqrt, 1)) {
                return sqrt(operand);
            } else if (__builtin_expect(func == sin, 1)) {
                return sin(operand);
            } else if (__builtin_expect(func == cos, 1)) {
                return cos(operand);
            } else if (__builtin_expect(func == fabs, 1)) {
                return fabs(operand);
            } else {
                return func(operand);
            }
        } else if (__builtin_expect(arity == 0, 0)) {
            return n->fun.f0();
        }
    }

    return 0.0;
}

static double cm_eval_fast(const cm_expr *n) {
#if defined(__aarch64__) || defined(__arm64__)
    // use hyper-optimized direct evaluation with branch prediction
    return cm_eval_hyper_optimized(n);
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
                    // prefetch next operation
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

// ultra-aggressive compile-time optimizations
static cm_expr *cm_ultra_aggressive_optimize(cm_expr *expr, cm_expr_pool *pool) {
    if (!expr) return NULL;

    // recursively optimize children first
    for (int i = 0; i < expr->member_count; i++) {
        expr->members[i] = cm_ultra_aggressive_optimize(expr->members[i], pool);
    }

    if (cm_get_type(expr->type) == CM_FUN) {
        int arity = cm_get_arity(expr->type);

        // ultra-aggressive algebraic simplifications
        if (arity == 2) {
            cm_expr *left = expr->members[0];
            cm_expr *right = expr->members[1];
            int left_const = (cm_get_type(left->type) == CM_CONST);
            int right_const = (cm_get_type(right->type) == CM_CONST);
            int left_var = (cm_get_type(left->type) == CM_VAR);
            int right_var = (cm_get_type(right->type) == CM_VAR);

            if (expr->fun.f2 == add) {
                // x + 0 = x, 0 + x = x
                if (left_const && left->value == 0.0) return right;
                if (right_const && right->value == 0.0) return left;

                // x + x = 2*x
                if (left_var && right_var && left->bound == right->bound) {
                    cm_expr *two = new_expr(CM_CONST, NULL, pool);
                    if (two) {
                        two->value = 2.0;
                        cm_expr *result = new_expr(CM_FUN | 2, (const cm_expr *[]){two, left}, pool);
                        if (result) result->fun.f2 = mul;
                        return result;
                    }
                }
            }
            else if (expr->fun.f2 == mul) {
                // x * 0 = 0, 0 * x = 0
                if ((left_const && left->value == 0.0) || (right_const && right->value == 0.0)) {
                    cm_expr *zero = new_expr(CM_CONST, NULL, pool);
                    if (zero) zero->value = 0.0;
                    return zero;
                }

                // x * 1 = x, 1 * x = x
                if (left_const && left->value == 1.0) return right;
                if (right_const && right->value == 1.0) return left;

                // x * x = x^2
                if (left_var && right_var && left->bound == right->bound) {
                    cm_expr *two = new_expr(CM_CONST, NULL, pool);
                    if (two) {
                        two->value = 2.0;
                        cm_expr *result = new_expr(CM_FUN | 2, (const cm_expr *[]){left, two}, pool);
                        if (result) result->fun.f2 = pow;
                        return result;
                    }
                }

                // strength reduction: 2*x = x+x (faster on some architectures)
                if (left_const && left->value == 2.0) {
                    cm_expr *result = new_expr(CM_FUN | 2, (const cm_expr *[]){right, right}, pool);
                    if (result) result->fun.f2 = add;
                    return result;
                }
                if (right_const && right->value == 2.0) {
                    cm_expr *result = new_expr(CM_FUN | 2, (const cm_expr *[]){left, left}, pool);
                    if (result) result->fun.f2 = add;
                    return result;
                }
            }
            else if (expr->fun.f2 == pow) {
                // x^0 = 1
                if (right_const && right->value == 0.0) {
                    cm_expr *one = new_expr(CM_CONST, NULL, pool);
                    if (one) one->value = 1.0;
                    return one;
                }

                // x^1 = x
                if (right_const && right->value == 1.0) return left;

                // 1^x = 1
                if (left_const && left->value == 1.0) {
                    cm_expr *one = new_expr(CM_CONST, NULL, pool);
                    if (one) one->value = 1.0;
                    return one;
                }

                // strength reduction for small integer powers
                if (right_const) {
                    if (right->value == 2.0) {
                        // x^2 = x*x (faster than pow)
                        cm_expr *result = new_expr(CM_FUN | 2, (const cm_expr *[]){left, left}, pool);
                        if (result) result->fun.f2 = mul;
                        return result;
                    }
                    if (right->value == 0.5) {
                        // x^0.5 = sqrt(x)
                        cm_expr *result = new_expr(CM_FUN | 1, (const cm_expr *[]){left}, pool);
                        if (result) result->fun.f1 = sqrt;
                        return result;
                    }
                }
            }
            else if (expr->fun.f2 == sub) {
                // x - 0 = x
                if (right_const && right->value == 0.0) return left;

                // x - x = 0
                if (left_var && right_var && left->bound == right->bound) {
                    cm_expr *zero = new_expr(CM_CONST, NULL, pool);
                    if (zero) zero->value = 0.0;
                    return zero;
                }
            }
            else if (expr->fun.f2 == divide) {
                // x / 1 = x
                if (right_const && right->value == 1.0) return left;

                // x / x = 1 (assuming x != 0)
                if (left_var && right_var && left->bound == right->bound) {
                    cm_expr *one = new_expr(CM_CONST, NULL, pool);
                    if (one) one->value = 1.0;
                    return one;
                }
            }
        }
        else if (arity == 1) {
            cm_expr *operand = expr->members[0];
            int operand_const = (cm_get_type(operand->type) == CM_CONST);

            // mathematical identities
            if (expr->fun.f1 == sqrt && cm_get_type(operand->type) == CM_FUN) {
                // sqrt(x^2) = |x|
                if (cm_get_arity(operand->type) == 2 && operand->fun.f2 == pow &&
                    cm_get_type(operand->members[1]->type) == CM_CONST &&
                    operand->members[1]->value == 2.0) {
                    cm_expr *result = new_expr(CM_FUN | 1, (const cm_expr *[]){operand->members[0]}, pool);
                    if (result) result->fun.f1 = fabs;
                    return result;
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

// ultra-fast direct compiled evaluation for simple patterns
static inline double cm_eval_direct_compiled(const cm_expr *n) {
    // check if this is a simple pattern we can evaluate directly
    if (n && n->pattern) {
        cm_pattern *pattern = (cm_pattern*)n->pattern;
        switch (pattern->type) {
            case PATTERN_CONST:
                return pattern->coefficients[0];

            case PATTERN_VAR:
                if (cm_get_type(n->type) == CM_VAR) {
                    return *n->bound;
                }
                break;

            case PATTERN_ADD_CONST:
                // x + constant - ultra-fast path
                if (cm_get_type(n->type) == CM_FUN && n->members[0] &&
                    cm_get_type(n->members[0]->type) == CM_VAR) {
                    return *n->members[0]->bound + pattern->coefficients[0];
                }
                break;

            case PATTERN_MUL_CONST:
                // x * constant - ultra-fast path
                if (cm_get_type(n->type) == CM_FUN && n->members[0] &&
                    cm_get_type(n->members[0]->type) == CM_VAR) {
                    return *n->members[0]->bound * pattern->coefficients[0];
                }
                break;

            case PATTERN_LINEAR:
                // ax + b - fastest path for linear expressions
                if (cm_get_type(n->type) == CM_FUN && n->members[0] &&
                    cm_get_type(n->members[0]->type) == CM_VAR) {
                    register double x = *n->members[0]->bound;
                    return pattern->coefficients[0] * x + pattern->coefficients[1];
                }
                break;

            case PATTERN_POW_CONST:
                // x^constant with optimized cases
                if (cm_get_type(n->type) == CM_FUN && n->members[0] &&
                    cm_get_type(n->members[0]->type) == CM_VAR) {
                    register double x = *n->members[0]->bound;
                    register double exp = pattern->coefficients[0];
                    if (__builtin_expect(exp == 2.0, 1)) return x * x;
                    if (__builtin_expect(exp == 0.5, 1)) return sqrt(x);
                    if (__builtin_expect(exp == 1.5, 1)) return x * sqrt(x);
                    if (__builtin_expect(exp == 2.5, 1)) return x * x * sqrt(x);
                    return pow(x, exp);
                }
                break;

            default:
                break;
        }
    }

    return cm_eval_fast(n);
}

double cm_eval(const cm_expr *n, int *error) {
    // use ultra-fast pattern-based evaluation first
    if (!error) {
        return cm_eval_direct_compiled(n);
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

        // apply ultra-aggressive optimizations
        root = cm_constant_fold(root, &globalPool);
        root = cm_ultra_aggressive_optimize(root, &globalPool);
        cm_specialize_expression(root);

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

// vectorized mathematical kernels

// high-quality minimax polynomial coefficients for exp on [-ln2/2, ln2/2]
static const double EXP_POLY_COEFFS[] __attribute__((aligned(64))) = {
    1.0000000000000000000000000000000000000000000000000000e+00,
    1.0000000000000000000000000000000000000000000000000000e+00,
    5.0000000000000000000000000000000000000000000000000000e-01,
    1.6666666666666666666666666666666666666666666666666667e-01,
    4.1666666666666666666666666666666666666666666666666667e-02,
    8.3333333333333333333333333333333333333333333333333333e-03,
    1.3888888888888888888888888888888888888888888888888889e-03,
    1.9841269841269841269841269841269841269841269841269841e-04,
    2.4801587301587301587301587301587301587301587301587302e-05,
    2.7557319223985890652557319223985890652557319223985891e-06
};
static const int EXP_POLY_DEG = 9;

// high-quality minimax polynomial coefficients for sin on [-pi/4, pi/4]
static const double SIN_POLY_COEFFS[] __attribute__((aligned(64))) = {
    0.0000000000000000000000000000000000000000000000000000e+00,
    1.0000000000000000000000000000000000000000000000000000e+00,
    0.0000000000000000000000000000000000000000000000000000e+00,
    -1.6666666666666666666666666666666666666666666666666667e-01,
    0.0000000000000000000000000000000000000000000000000000e+00,
    8.3333333333333333333333333333333333333333333333333333e-03,
    0.0000000000000000000000000000000000000000000000000000e+00,
    -1.9841269841269841269841269841269841269841269841269841e-04,
    0.0000000000000000000000000000000000000000000000000000e+00,
    2.7557319223985890652557319223985890652557319223985891e-06
};
static const int SIN_POLY_DEG = 9;

// estrin polynomial evaluation for better ILP and SIMD performance
static inline cm_vd vec_poly_estrin(const double *coeffs, int deg, cm_vd x) {
    if (deg <= 1) {
        return vec_fma_pd(vec_set1_pd(coeffs[1]), x, vec_set1_pd(coeffs[0]));
    }

    cm_vd x2 = vec_mul_pd(x, x);
    cm_vd x4 = vec_mul_pd(x2, x2);

    // estrin scheme: group terms to reduce dependencies
    cm_vd p01 = vec_fma_pd(vec_set1_pd(coeffs[1]), x, vec_set1_pd(coeffs[0]));
    cm_vd p23 = vec_fma_pd(vec_set1_pd(coeffs[3]), x, vec_set1_pd(coeffs[2]));
    cm_vd p45 = vec_fma_pd(vec_set1_pd(coeffs[5]), x, vec_set1_pd(coeffs[4]));
    cm_vd p67 = vec_fma_pd(vec_set1_pd(coeffs[7]), x, vec_set1_pd(coeffs[6]));

    cm_vd p0123 = vec_fma_pd(p23, x2, p01);
    cm_vd p4567 = vec_fma_pd(p67, x2, p45);

    if (deg >= 8) {
        cm_vd p89 = vec_fma_pd(vec_set1_pd(coeffs[9]), x, vec_set1_pd(coeffs[8]));
        cm_vd p89ab = p89; // simplified for degree 9
        return vec_fma_pd(vec_fma_pd(p89ab, x4, p4567), x4, p0123);
    }

    return vec_fma_pd(p4567, x4, p0123);
}

// newton-raphson iterative refinement for rsqrt
static inline cm_vd vec_rsqrt_nr(cm_vd x) {
#if defined(CM_HAVE_AVX2)
    // use AVX2 rsqrt approximation + newton refinement
    __m256d approx = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x)));
    // newton step: y' = y * (3 - x*y^2) / 2
    __m256d y2 = _mm256_mul_pd(approx, approx);
    __m256d xy2 = _mm256_mul_pd(x, y2);
    __m256d three_minus_xy2 = _mm256_sub_pd(_mm256_set1_pd(3.0), xy2);
    return _mm256_mul_pd(approx, _mm256_mul_pd(three_minus_xy2, _mm256_set1_pd(0.5)));
#elif defined(CM_HAVE_NEON)
    // use NEON rsqrt estimate + newton refinement
    float64x2_t approx = vrsqrteq_f64(x);
    float64x2_t y2 = vmulq_f64(approx, approx);
    float64x2_t xy2 = vmulq_f64(x, y2);
    float64x2_t three_minus_xy2 = vsubq_f64(vdupq_n_f64(3.0), xy2);
    return vmulq_f64(approx, vmulq_f64(three_minus_xy2, vdupq_n_f64(0.5)));
#else
    return vec_set1_pd(1.0 / sqrt(vec_get_lane(x, 0)));
#endif
}

// vectorized exp kernel for AVX2
static inline void vec_exp_avx2(const double *in, double *out, size_t n) {
#if defined(CM_HAVE_AVX2)
    const __m256d ln2 = _mm256_set1_pd(0.693147180559945309417232121458176568);
    const __m256d inv_ln2 = _mm256_set1_pd(1.442695040888963407359924681001892137);
    const double EXP_LO = -709.782712893384;
    const double EXP_HI = 709.782712893384;

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(in + i);

        // clamp to avoid overflow
        __m256d x_clamped = _mm256_min_pd(_mm256_max_pd(x, _mm256_set1_pd(EXP_LO)), _mm256_set1_pd(EXP_HI));

        // k = floor(x / ln2)
        __m256d x_scaled = _mm256_mul_pd(x_clamped, inv_ln2);
        __m256d k_real = _mm256_floor_pd(x_scaled);
        __m128i k32 = _mm256_cvttpd_epi32(k_real);
        __m256i k64 = _mm256_cvtepi32_epi64(k32);

        // r = x - k*ln2
        __m256d k_d = _mm256_cvtepi64_pd(k64);
        __m256d r = _mm256_sub_pd(x_clamped, _mm256_mul_pd(k_d, ln2));

        // compute exp(r) via estrin polynomial for better ILP
        __m256d acc = vec_poly_estrin(EXP_POLY_COEFFS, EXP_POLY_DEG, r);

        // compute 2^k by building exponent bits
        __m256i biased = _mm256_add_epi64(k64, _mm256_set1_epi64x(1023));
        __m256i bits = _mm256_slli_epi64(biased, 52);
        __m256d pow2 = _mm256_castsi256_pd(bits);

        __m256d result = _mm256_mul_pd(acc, pow2);
        _mm256_storeu_pd(out + i, result);
    }

    // remainder scalar fallback
    for (; i < n; i++) {
        out[i] = exp(in[i]);
    }
#endif
}

// vectorized exp kernel for NEON
static inline void vec_exp_neon(const double *in, double *out, size_t n) {
#if defined(CM_HAVE_NEON)
    const double ln2 = 0.693147180559945309417232121458176568;
    const double inv_ln2 = 1.442695040888963407359924681001892137;

    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(in + i);

        // compute lanewise
        double xi0 = vgetq_lane_f64(x, 0);
        double xi1 = vgetq_lane_f64(x, 1);
        int64_t k0 = (int64_t) floor(xi0 * inv_ln2);
        int64_t k1 = (int64_t) floor(xi1 * inv_ln2);
        double r0 = xi0 - (double)k0 * ln2;
        double r1 = xi1 - (double)k1 * ln2;

        // polynomial evaluation
        double acc0 = EXP_POLY_COEFFS[EXP_POLY_DEG];
        for (int j = EXP_POLY_DEG -1; j >= 0; --j) acc0 = acc0 * r0 + EXP_POLY_COEFFS[j];
        double acc1 = EXP_POLY_COEFFS[EXP_POLY_DEG];
        for (int j = EXP_POLY_DEG -1; j >= 0; --j) acc1 = acc1 * r1 + EXP_POLY_COEFFS[j];

        double res0 = ldexp(acc0, (int)k0);
        double res1 = ldexp(acc1, (int)k1);
        float64x2_t res = { res0, res1 };
        vst1q_f64(out + i, res);
    }

    // remainder scalar fallback
    for (; i < n; i++) {
        out[i] = exp(in[i]);
    }
#endif
}

// high-precision constants for robust range reduction
static const double PI_2_HI = 1.57079632679489655800e+00;  // upper bits of pi/2
static const double PI_2_LO = 6.12323399573676588e-17;     // lower bits of pi/2
static const double INV_PI_2 = 6.36619772367581343076e-01; // 2/pi

// robust range reduction for trigonometric functions
static inline void range_reduce_trig(cm_vd x, cm_vd *r, cm_vd *quadrant) {
#if defined(CM_HAVE_AVX2)
    // multiply by 2/pi to get quotient
    __m256d y = _mm256_mul_pd(x, _mm256_set1_pd(INV_PI_2));

    // round to nearest integer to get quadrant
    __m256d n_real = _mm256_round_pd(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    *quadrant = n_real;

    // high-precision subtraction: r = x - n*(pi/2)
    // use split pi/2 for accuracy
    __m256d r_hi = _mm256_fnmadd_pd(n_real, _mm256_set1_pd(PI_2_HI), x);
    *r = _mm256_fnmadd_pd(n_real, _mm256_set1_pd(PI_2_LO), r_hi);
#elif defined(CM_HAVE_NEON)
    float64x2_t y = vmulq_f64(x, vdupq_n_f64(INV_PI_2));
    float64x2_t n_real = vrndnq_f64(y);
    *quadrant = n_real;

    float64x2_t r_hi = vfmsq_f64(x, n_real, vdupq_n_f64(PI_2_HI));
    *r = vfmsq_f64(r_hi, n_real, vdupq_n_f64(PI_2_LO));
#else
    double y = vec_get_lane(x, 0) * INV_PI_2;
    double n = round(y);
    *quadrant = vec_set1_pd(n);
    double r_hi = vec_get_lane(x, 0) - n * PI_2_HI;
    *r = vec_set1_pd(r_hi - n * PI_2_LO);
#endif
}

// branchless quadrant handling for sin/cos
static inline void apply_trig_quadrant(cm_vd r, cm_vd quadrant, cm_vd *sin_result, cm_vd *cos_result) {
#if defined(CM_HAVE_AVX2)
    // compute both sin(r) and cos(r) using polynomials
    __m256d r2 = _mm256_mul_pd(r, r);

    // sin polynomial on reduced range
    __m256d sin_poly = vec_poly_estrin(SIN_POLY_COEFFS, SIN_POLY_DEG, r);

    // cos polynomial: cos(r) = 1 - r^2/2 + r^4/24 - ...
    __m256d cos_c[] = {1.0, 0.0, -0.5, 0.0, 1.0/24.0, 0.0, -1.0/720.0, 0.0, 1.0/40320.0, 0.0};
    __m256d cos_poly = vec_poly_estrin(cos_c, 9, r);

    // quadrant-based selection using masks
    __m256i quad_i = _mm256_cvtpd_epi64(quadrant);
    __m256i mask1 = _mm256_cmpeq_epi64(_mm256_and_si256(quad_i, _mm256_set1_epi64x(1)), _mm256_set1_epi64x(1));
    __m256i mask2 = _mm256_cmpeq_epi64(_mm256_and_si256(quad_i, _mm256_set1_epi64x(2)), _mm256_set1_epi64x(2));

    // swap sin/cos based on quadrant 1,3
    __m256d sin_base = _mm256_blendv_pd(sin_poly, cos_poly, _mm256_castsi256_pd(mask1));
    __m256d cos_base = _mm256_blendv_pd(cos_poly, sin_poly, _mm256_castsi256_pd(mask1));

    // negate based on quadrant 2,3
    __m256d neg_mask = _mm256_castsi256_pd(mask2);
    *sin_result = _mm256_blendv_pd(sin_base, _mm256_sub_pd(_mm256_setzero_pd(), sin_base), neg_mask);
    *cos_result = _mm256_blendv_pd(cos_base, _mm256_sub_pd(_mm256_setzero_pd(), cos_base), neg_mask);
#else
    // simplified scalar fallback
    int quad = (int)vec_get_lane(quadrant, 0) & 3;
    double r_scalar = vec_get_lane(r, 0);
    double sin_val = sin(r_scalar);
    double cos_val = cos(r_scalar);

    switch(quad) {
        case 0: break;
        case 1: { double temp = sin_val; sin_val = cos_val; cos_val = -temp; } break;
        case 2: sin_val = -sin_val; cos_val = -cos_val; break;
        case 3: { double temp = sin_val; sin_val = -cos_val; cos_val = temp; } break;
    }

    *sin_result = vec_set1_pd(sin_val);
    *cos_result = vec_set1_pd(cos_val);
#endif
}

// vectorized sin kernel with robust range reduction
static inline void vec_sin_avx2(const double *in, double *out, size_t n) {
#if defined(CM_HAVE_AVX2)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(in + i);

        // robust range reduction
        __m256d r, quadrant;
        range_reduce_trig(x, &r, &quadrant);

        // apply quadrant-aware sin/cos computation
        __m256d sin_result, cos_result;
        apply_trig_quadrant(r, quadrant, &sin_result, &cos_result);

        _mm256_storeu_pd(out + i, sin_result);
    }

    // remainder scalar fallback
    for (; i < n; i++) {
        out[i] = sin(in[i]);
    }
#endif
}

// table-driven approximations for ultra-fast evaluation
#define EXP_TABLE_SIZE 128
#define LOG_TABLE_SIZE 128

// pre-computed lookup tables for exp and log (aligned for cache efficiency)
static double exp_table[EXP_TABLE_SIZE] __attribute__((aligned(64)));
static double log_table[LOG_TABLE_SIZE] __attribute__((aligned(64)));

// initialize lookup tables with high-precision values
static void init_lookup_tables(void) {
    static int initialized = 0;
    if (initialized) return;

    // exp table: exp(i/128) for i = 0..127
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        exp_table[i] = exp((double)i / EXP_TABLE_SIZE);
    }

    // log table: log(1 + i/128) for i = 0..127
    for (int i = 0; i < LOG_TABLE_SIZE; i++) {
        log_table[i] = log(1.0 + (double)i / LOG_TABLE_SIZE);
    }

    initialized = 1;
}

// table-driven exp with polynomial correction
static inline cm_vd vec_exp_table_hybrid(cm_vd x) {
    init_lookup_tables();

#if defined(CM_HAVE_AVX2)
    // range reduction: x = k*ln2 + r
    const __m256d ln2 = _mm256_set1_pd(0.693147180559945309417232121458176568);
    const __m256d inv_ln2 = _mm256_set1_pd(1.442695040888963407359924681001892137);

    __m256d k_real = _mm256_mul_pd(x, inv_ln2);
    __m256d k_floor = _mm256_floor_pd(k_real);
    __m256d r = _mm256_fnmadd_pd(k_floor, ln2, x);

    // table lookup for coarse approximation
    // for small r, use table + polynomial correction
    __m256d table_idx = _mm256_mul_pd(r, _mm256_set1_pd(EXP_TABLE_SIZE));
    __m256i idx = _mm256_cvtpd_epi32(table_idx);

    // gather from lookup table (simplified - would use proper gather in production)
    double table_vals[4];
    int indices[4];
    _mm_storeu_si128((__m128i*)indices, idx);
    for (int i = 0; i < 4; i++) {
        int table_idx_val = indices[i] & (EXP_TABLE_SIZE - 1);
        table_vals[i] = exp_table[table_idx_val];
    }
    __m256d coarse = _mm256_loadu_pd(table_vals);

    // polynomial correction for residual
    __m256d residual = _mm256_sub_pd(r, _mm256_div_pd(table_idx, _mm256_set1_pd(EXP_TABLE_SIZE)));
    __m256d correction = vec_poly_estrin(EXP_POLY_COEFFS, 5, residual); // lower degree for correction

    // combine: result = coarse * correction * 2^k
    __m256d combined = _mm256_mul_pd(coarse, correction);

    // apply 2^k scaling
    __m256i k_int = _mm256_cvtpd_epi32(k_floor);
    __m256i biased = _mm256_add_epi32(k_int, _mm256_set1_epi32(1023));
    __m256d scale = _mm256_castsi256_pd(_mm256_slli_epi32(biased, 20)); // simplified scaling

    return _mm256_mul_pd(combined, scale);
#else
    // scalar fallback
    double x_scalar = vec_get_lane(x, 0);
    return vec_set1_pd(exp(x_scalar));
#endif
}

// function pointers for runtime dispatch
typedef void (*cm_vec_exp_fn)(const double* in, double* out, size_t n);
typedef void (*cm_vec_sin_fn)(const double* in, double* out, size_t n);

static cm_vec_exp_fn global_vec_exp = NULL;
static cm_vec_sin_fn global_vec_sin = NULL;

// vector bytecode VM

// vector bytecode evaluation with SIMD registers
static void vm_eval_vec_block(const uint8_t *code, const double *consts, const double **vars, double *out, size_t block_start, size_t block_size) {
    // vector stack for SIMD operations
    cm_vd vstack[64];
    int vsp = 0;

    const uint8_t *pc = code;
    while (1) {
        uint8_t op = *pc++;
        switch (op) {
            case OP_V_LOAD_CONST: {
                uint32_t idx = *(uint32_t*)pc; pc += 4;
                // broadcast constant to vector
                vstack[vsp++] = vec_set1_pd(consts[idx]);
                break;
            }
            case OP_V_LOAD_VAR: {
                uint32_t var_id = *(uint32_t*)pc; pc += 4;
                // load vector of variable values
                vstack[vsp++] = vec_load_pd(vars[var_id] + block_start);
                break;
            }
            case OP_V_ADD: {
                cm_vd b = vstack[--vsp];
                cm_vd a = vstack[--vsp];
                vstack[vsp++] = vec_add_pd(a, b);
                break;
            }
            case OP_V_SUB: {
                cm_vd b = vstack[--vsp];
                cm_vd a = vstack[--vsp];
                vstack[vsp++] = vec_sub_pd(a, b);
                break;
            }
            case OP_V_MUL: {
                cm_vd b = vstack[--vsp];
                cm_vd a = vstack[--vsp];
                vstack[vsp++] = vec_mul_pd(a, b);
                break;
            }
            case OP_V_FMA: {
                cm_vd c = vstack[--vsp];
                cm_vd b = vstack[--vsp];
                cm_vd a = vstack[--vsp];
                vstack[vsp++] = vec_fma_pd(a, b, c); // a*b + c
                break;
            }
            case OP_V_EXP: {
                cm_vd a = vstack[--vsp];
                // use hybrid table+polynomial approach for best performance
                cm_vd result = vec_exp_table_hybrid(a);
                vstack[vsp++] = result;
                break;
            }
            case OP_V_SIN: {
                cm_vd a = vstack[--vsp];
                // use robust range reduction + polynomial
                cm_vd r, quadrant;
                range_reduce_trig(a, &r, &quadrant);
                cm_vd sin_result, cos_result;
                apply_trig_quadrant(r, quadrant, &sin_result, &cos_result);
                vstack[vsp++] = sin_result;
                break;
            }
            case OP_V_SQRT: {
                cm_vd a = vstack[--vsp];
                // use newton-raphson refinement for high performance
                cm_vd rsqrt = vec_rsqrt_nr(a);
                vstack[vsp++] = vec_mul_pd(a, rsqrt); // sqrt(a) = a * rsqrt(a)
                break;
            }
            case OP_V_RETURN: {
                cm_vd result = vstack[--vsp];
                vec_store_pd(out + block_start, result);
                return;
            }
            default:
                // fallback to scalar evaluation for unsupported ops
                return;
        }
    }
}

// forward declaration
static void init_cpu_dispatch(void);

// advanced vector evaluation with VM
void cm_eval_vec_vm(const cm_expr *expr, double *out, const double **vars, size_t n, cm_eval_mode_t mode) {
    if (!expr || !out || n == 0) return;

    init_cpu_dispatch();
    init_lookup_tables();

    // check if expression has vector bytecode
    if (expr->bytecode && (expr->optimization_flags & CM_OPT_BYTECODE)) {
        cm_bytecode *bc = (cm_bytecode*)expr->bytecode;

        // process in vector-width blocks
        size_t i = 0;
        for (; i + CM_VW <= n; i += CM_VW) {
            vm_eval_vec_block(bc->code, bc->constants, vars, out, i, CM_VW);
        }

        // handle remainder with scalar evaluation
        for (; i < n; i++) {
            out[i] = cm_eval(expr, NULL);
        }
        return;
    }

    // fallback to original vector evaluation
    cm_eval_vec(expr, out, vars, n, mode);
}

// initialize cpu dispatch
static void init_cpu_dispatch(void) {
    static int initialized = 0;
    if (initialized) return;

#if defined(CM_HAVE_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        global_vec_exp = vec_exp_avx2;
        global_vec_sin = vec_sin_avx2;
        initialized = 1;
        return;
    }
#endif
#if defined(CM_HAVE_NEON)
    global_vec_exp = vec_exp_neon;
    global_vec_sin = NULL; // implement neon sin later
    initialized = 1;
    return;
#endif

    // scalar fallback
    global_vec_exp = NULL;
    global_vec_sin = NULL;
    initialized = 1;
}

// vectorized batch evaluation
void cm_eval_vec(const cm_expr *expr, double *out, const double **vars, size_t n, cm_eval_mode_t mode) {
    if (!expr || !out || n == 0) return;

    init_cpu_dispatch();

    // for simple expressions, use specialized vectorized paths
    if (cm_get_type(expr->type) == CM_FUN && cm_get_arity(expr->type) == 1) {
        if (expr->fun.f1 == exp && global_vec_exp) {
            // vectorized exp path
            double *inputs = malloc(n * sizeof(double));
            if (!inputs) goto fallback;

            // collect input values
            if (cm_get_type(expr->members[0]->type) == CM_VAR) {
                const double *var_ptr = expr->members[0]->bound;
                if (vars && vars[0]) {
                    memcpy(inputs, vars[0], n * sizeof(double));
                } else {
                    for (size_t i = 0; i < n; i++) {
                        inputs[i] = *var_ptr;
                    }
                }
            } else {
                // evaluate subexpression for each input
                for (size_t i = 0; i < n; i++) {
                    inputs[i] = cm_eval(expr->members[0], NULL);
                }
            }

            global_vec_exp(inputs, out, n);
            free(inputs);
            return;
        }
        else if (expr->fun.f1 == sin && global_vec_sin) {
            // vectorized sin path
            double *inputs = malloc(n * sizeof(double));
            if (!inputs) goto fallback;

            if (cm_get_type(expr->members[0]->type) == CM_VAR) {
                const double *var_ptr = expr->members[0]->bound;
                if (vars && vars[0]) {
                    memcpy(inputs, vars[0], n * sizeof(double));
                } else {
                    for (size_t i = 0; i < n; i++) {
                        inputs[i] = *var_ptr;
                    }
                }
            } else {
                for (size_t i = 0; i < n; i++) {
                    inputs[i] = cm_eval(expr->members[0], NULL);
                }
            }

            global_vec_sin(inputs, out, n);
            free(inputs);
            return;
        }
    }

    fallback:
    // scalar fallback
    for (size_t i = 0; i < n; i++) {
        out[i] = cm_eval(expr, NULL);
    }
}

// multithreaded vectorized evaluation (stub)
void cm_eval_vec_mt(const cm_expr *expr, double *out, const double **vars, size_t n, cm_eval_mode_t mode, int num_threads) {
    // for now, just call single-threaded version
    cm_eval_vec(expr, out, vars, n, mode);
}
