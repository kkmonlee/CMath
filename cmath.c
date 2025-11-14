#include "cmath.h"

#include <math.h>
#include <stdlib.h>

#define CM_INIT_INSTR_CAP 64

struct cm_program {
    size_t num_vars;
    size_t num_slots;
    uint32_t result_slot;
    int result_set;

    cm_instr *code;
    size_t n_insts;
    size_t inst_cap;

    // Scalar JIT
    cm_jit_fn jit_fn;
    void *jit_state;
    int jitted;

    // Batch JIT
    cm_jit_fn_batch jit_fn_batch;
    void *jit_state_batch;
    int jitted_batch;

    // Interpreter scratch
    double *scratch;
    size_t scratch_cap;
};

// FIXME: change buffer cap to > 1
static int cm_ensure_capacity(cm_program *p, size_t extra) {
    if (p->n_insts + extra <= p->inst_cap) return 0;
    size_t new_cap = p->inst_cap ? p->inst_cap : CM_INIT_INSTR_CAP;
    while (p->n_insts + extra > new_cap) new_cap *= 2;
    cm_instr *nbuf = realloc(p->code, new_cap * sizeof(cm_instr));
    if (!nbuf) return 1;
    p->code = nbuf;
    p->inst_cap = new_cap;
    return 0;
}

static uint32_t cm_alloc_slot(cm_program *p) {
    const uint32_t s = (uint32_t) p->num_slots;
    p->num_slots++;
    return s;
}

static uint32_t cm_emit_raw(cm_program *p, cm_instr ins) {
    if (cm_ensure_capacity(p, 1)) return UINT32_MAX;
    p->code[p->n_insts++] = ins;
    if (!p->result_set) p->result_slot = ins.dst;
    return ins.dst;
}

// API impl (emission)
uint32_t cm_emit_const(cm_program *p, double value) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_CONST;
    i.dst = d;
    i.imm = value;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_var(cm_program *p, uint32_t vi) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_VAR;
    i.dst = d;
    i.aux = (int32_t) vi;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_neg(cm_program *p, uint32_t a) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_NEG;
    i.dst = d;
    i.a = a;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_sqrt(cm_program *p, uint32_t a) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_SQRT;
    i.dst = d;
    i.a = a;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_recip(cm_program *p, uint32_t a) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_RECIP;
    i.dst = d;
    i.a = a;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_powi(cm_program *p, uint32_t a, int e) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_POWI;
    i.dst = d;
    i.a = a;
    if (e > 8)e = 8;
    if (e < -8)e = -8;
    i.aux = e;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_add(cm_program *p, uint32_t a, uint32_t b) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_ADD;
    i.dst = d;
    i.a = a;
    i.b = b;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_sub(cm_program *p, uint32_t a, uint32_t b) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_SUB;
    i.dst = d;
    i.a = a;
    i.b = b;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_mul(cm_program *p, uint32_t a, uint32_t b) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_MUL;
    i.dst = d;
    i.a = a;
    i.b = b;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_div(cm_program *p, uint32_t a, uint32_t b) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_DIV;
    i.dst = d;
    i.a = a;
    i.b = b;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_add_k(cm_program *p, uint32_t a, double k) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_ADD_K;
    i.dst = d;
    i.a = a;
    i.imm = k;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_mul_k(cm_program *p, uint32_t a, double k) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_MUL_K;
    i.dst = d;
    i.a = a;
    i.imm = k;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_fma(cm_program *p, uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_FMA;
    i.dst = d;
    i.a = a;
    i.b = b;
    i.c = c;
    return cm_emit_raw(p, i);
}

uint32_t cm_emit_abs(cm_program *p, uint32_t a) {
    const uint32_t d = cm_alloc_slot(p);
    cm_instr i = {0};
    i.op = CM_OP_ABS;
    i.dst = d;
    i.a = a;
    return cm_emit_raw(p, i);
}

cm_program *cm_prog_create(size_t num_vars, size_t initial_slots) {
    cm_program *p = calloc(1, sizeof(cm_program));
    if (!p) return NULL;
    p->num_vars = num_vars;
    p->num_slots = 0;
    p->result_slot = 0;
    p->result_set = 0;
    p->code = NULL;
    p->n_insts = 0;
    p->inst_cap = 0;
    p->jit_fn = NULL;
    p->jit_state = NULL;
    p->jitted = 0;
    p->jit_fn_batch = NULL;
    p->jit_state_batch = NULL;
    p->jitted_batch = 0;
    p->scratch = NULL;
    p->scratch_cap = 0;
    for (size_t i = 0; i < initial_slots; ++i) (void) cm_alloc_slot(p);
    return p;
}

void cm_prog_free(cm_program *p) {
    if (!p) return;
    if (p->jit_state) {
        cm_llvm_jit_release(p->jit_state);
        p->jit_state = NULL;
        p->jit_fn = NULL;
        p->jitted = 0;
    }
    if (p->jit_state_batch) {
        cm_llvm_jit_release(p->jit_state_batch);
        p->jit_state_batch = NULL;
        p->jit_fn_batch = NULL;
        p->jitted_batch = 0;
    }
    free(p->scratch);
    free(p->code);
    free(p);
}

void cm_set_result(cm_program *p, uint32_t result_slot) {
    if (!p) return;
    p->result_slot = result_slot;
    p->result_set = 1;
}

int cm_jit_globally_supported(void) { return cm_llvm_jit_supported(); }
int cm_prog_is_jitted(const cm_program *p) { return p ? p->jitted : 0; }
int cm_prog_is_jitted_batch(const cm_program *p) { return p ? p->jitted_batch : 0; }

// interpreter (scalar) with reusable scratch
static double cm_eval_interpreter(cm_program *p, const double *vars) {
    if (p->num_slots == 0) return 0.0;
    if (p->scratch_cap < p->num_slots) {
        size_t new_cap = p->num_slots;
        if (p->scratch_cap > 0) {
            new_cap = (p->scratch_cap * 3) / 2;
            if (new_cap < p->num_slots) new_cap = p->num_slots;
        }
        double *buf = realloc(p->scratch, new_cap * sizeof(double));
        if (!buf) return 0.0;
        p->scratch = buf;
        p->scratch_cap = new_cap;
    }
    double *slots = p->scratch;
    for (size_t i = 0; i < p->n_insts; ++i) {
        const cm_instr *ins = &p->code[i];
        switch ((cm_opcode) ins->op) {
            case CM_OP_CONST: slots[ins->dst] = ins->imm;
                break;
            case CM_OP_VAR: slots[ins->dst] = ((uint32_t) ins->aux < p->num_vars) ? vars[(uint32_t) ins->aux] : 0.0;
                break;
            case CM_OP_ADD: slots[ins->dst] = slots[ins->a] + slots[ins->b];
                break;
            case CM_OP_SUB: slots[ins->dst] = slots[ins->a] - slots[ins->b];
                break;
            case CM_OP_MUL: slots[ins->dst] = slots[ins->a] * slots[ins->b];
                break;
            case CM_OP_DIV: slots[ins->dst] = slots[ins->a] / slots[ins->b];
                break;
            case CM_OP_NEG: slots[ins->dst] = -slots[ins->a];
                break;
            case CM_OP_SQRT: slots[ins->dst] = sqrt(slots[ins->a]);
                break;
            case CM_OP_ADD_K: slots[ins->dst] = slots[ins->a] + ins->imm;
                break;
            case CM_OP_MUL_K: slots[ins->dst] = slots[ins->a] * ins->imm;
                break;
            case CM_OP_RECIP: slots[ins->dst] = 1.0 / slots[ins->a];
                break;
            case CM_OP_ABS: slots[ins->dst] = fabs(slots[ins->a]);
                break;
            case CM_OP_POWI: {
                int e = ins->aux;
                if (e > 8) e = 8;
                if (e < -8) e = -8;
                double x = slots[ins->a], r;
                if (e == 0) r = 1.0;
                else if (e == 1) r = x;
                else if (e == -1) r = 1.0 / x;
                else {
                    const int neg = e < 0;
                    const unsigned k = (unsigned) (neg ? -e : e);
                    const double xx = x * x;
                    switch (k) {
                        case 2: r = xx;
                            break;
                        case 3: r = xx * x;
                            break;
                        case 4: r = xx * xx;
                            break;
                        case 5: r = xx * xx * x;
                            break;
                        case 6: r = xx * xx * xx;
                            break;
                        case 7: r = xx * xx * xx * x;
                            break;
                        case 8: {
                            const double xxxx = xx * xx;
                            r = xxxx * xxxx;
                            break;
                        }
                        default: r = xx;
                            for (unsigned j = 2; j < k; ++j) r *= x;
                            break;
                    }
                    if (neg) r = 1.0 / r;
                }
                slots[ins->dst] = r;
                break;
            }
            case CM_OP_FMA: slots[ins->dst] = fma(slots[ins->a], slots[ins->b], slots[ins->c]);
                break;
            default: slots[ins->dst] = 0.0;
                break;
        }
    }
    return p->result_slot < p->num_slots ? slots[p->result_slot] : 0.0;
}

static int cm_compile_inner(cm_program *p, const cm_jit_options *opts) {
    if (!p) return 1;
    if (p->n_insts == 0) {
        (void) cm_emit_const(p, 0.0);
        if (!p->result_set) p->result_slot = p->code[p->n_insts - 1].dst;
    }
    if (p->result_slot >= p->num_slots) p->result_slot = p->num_slots ? (uint32_t) (p->num_slots - 1) : 0u;
    p->jitted = 0;
    if (cm_llvm_jit_supported()) {
        cm_jit_fn fn = NULL;
        void *st = NULL;
        const cm_jit_options local = *opts;
        const int rc = cm_llvm_jit_compile_ex(p->code, p->n_insts, p->num_vars, p->num_slots, p->result_slot,
                                        &local, &fn, &st);
        if (rc == 0 && fn) {
            p->jit_fn = fn;
            p->jit_state = st;
            p->jitted = 1;
        }
    }
    return 0;
}

int cm_compile(cm_program *p) {
    const cm_jit_options o = {
        .opt_level = 3, .enable_const_fold = 1, .enable_cse = 1, .enable_dce = 1, .enable_auto_fma = 1, .powi_limit = 8,
        .vec_width_hint = 0, .interleave_hint = 2, .unroll_hint = 4, .alignment = 16, .prefetch_distance = 64,
        .block_size = 0,
        .assume_noalias = 1
    };
    return cm_compile_inner(p, &o);
}

int cm_compile_ex(cm_program *p, const cm_jit_options *opts) { return cm_compile_inner(p, opts); }

double cm_eval(const cm_program *pconst, const double *vars) {
    cm_program *p = (cm_program *) pconst;
    if (!p) return 0.0;
    if (p->jitted && p->jit_fn) return p->jit_fn(vars);
    return cm_eval_interpreter(p, vars);
}

// batch compile/eval
static int cm_compile_batch_inner(cm_program *p, const cm_jit_options *opts) {
    if (!p) return 1;
    if (p->n_insts == 0) {
        (void) cm_emit_const(p, 0.0);
        if (!p->result_set) p->result_slot = p->code[p->n_insts - 1].dst;
    }
    if (p->result_slot >= p->num_slots) p->result_slot = p->num_slots ? (uint32_t) (p->num_slots - 1) : 0u;
    p->jitted_batch = 0;
    if (cm_llvm_jit_supported()) {
        cm_jit_fn_batch fn = NULL;
        void *st = NULL;
        const cm_jit_options local = *opts;
        const int rc = cm_llvm_jit_compile_batch_ex(p->code, p->n_insts, p->num_vars, p->num_slots, p->result_slot,
                                              &local, &fn, &st);
        if (rc == 0 && fn) {
            p->jit_fn_batch = fn;
            p->jit_state_batch = st;
            p->jitted_batch = 1;
        }
    }
    return 0;
}

int cm_compile_batch(cm_program *p) {
    const cm_jit_options o = {
        .opt_level = 3, .enable_const_fold = 1, .enable_cse = 1, .enable_dce = 1, .enable_auto_fma = 1, .powi_limit = 8,
        .vec_width_hint = 0, .interleave_hint = 4, .unroll_hint = 4, .alignment = 64, .prefetch_distance = 128,
        .block_size = 0,
        .assume_noalias = 1
    };
    return cm_compile_batch_inner(p, &o);
}

int cm_compile_batch_ex(cm_program *p, const cm_jit_options *opts) { return cm_compile_batch_inner(p, opts); }

// portable vectorized C fallback for batch (SoA)
void cm_eval_batch(const cm_program *pconst,
                   const double *const*inputs, size_t n, double *out) {
    cm_program *p = (cm_program *) pconst;
    if (!p) return;

    // use JIT batch if present
    if (p->jitted_batch && p->jit_fn_batch) {
        p->jit_fn_batch(inputs, n, out);
        return;
    }

    // otherwise evaluate per element using the existing scalar path.
    // we avoid per-iter allocations by using a small fixed local buffer,
    // and only heap-alloc if num_vars exceeds the fixed size.
    const size_t NV = p->num_vars;
    double small_buf[16];
    double *vbuf = small_buf;
    if (NV > (sizeof(small_buf) / sizeof(small_buf[0]))) {
        vbuf = (double *) malloc(NV * sizeof(double));
        if (!vbuf) return; // give up quietly
    }

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#pragma clang loop vectorize_width(4) interleave_count(2)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (size_t i = 0; i < n; ++i) {
#if defined(__clang__)
        // Prefetch next iteration
        if (i + 64 < n) {
            __builtin_prefetch((const void *) &out[i + 64], 1, 3);
            for (size_t j = 0; j < NV; ++j)
                __builtin_prefetch((const void *) (inputs[j] + i + 64), 0, 3);
        }
#endif
        // Gather SoA -> small AoS
        for (size_t j = 0; j < NV; ++j)
            vbuf[j] = inputs[j][i];

        out[i] = cm_eval(p, vbuf);
    }

    if (vbuf != small_buf) free(vbuf);
}
