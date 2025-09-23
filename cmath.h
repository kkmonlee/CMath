#ifndef CMATH_H
#define CMATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "cmath_jit_llvm.h"

    // Opaque program
    typedef struct cm_program cm_program;

    // Create / destroy
    cm_program* cm_prog_create(size_t num_vars, size_t initial_slots);
    void        cm_prog_free(cm_program* p);

    // Emit
    uint32_t cm_emit_const(cm_program* p, double value);
    uint32_t cm_emit_var(cm_program* p, uint32_t var_index);
    uint32_t cm_emit_neg(cm_program* p, uint32_t a);
    uint32_t cm_emit_sqrt(cm_program* p, uint32_t a);
    uint32_t cm_emit_recip(cm_program* p, uint32_t a);
    uint32_t cm_emit_powi(cm_program* p, uint32_t a, int exponent);
    uint32_t cm_emit_add(cm_program* p, uint32_t a, uint32_t b);
    uint32_t cm_emit_sub(cm_program* p, uint32_t a, uint32_t b);
    uint32_t cm_emit_mul(cm_program* p, uint32_t a, uint32_t b);
    uint32_t cm_emit_div(cm_program* p, uint32_t a, uint32_t b);
    uint32_t cm_emit_add_k(cm_program* p, uint32_t a, double k);
    uint32_t cm_emit_mul_k(cm_program* p, uint32_t a, double k);
    uint32_t cm_emit_fma(cm_program* p, uint32_t a, uint32_t b, uint32_t c);
    uint32_t cm_emit_abs(cm_program* p, uint32_t a);

    void     cm_set_result(cm_program* p, uint32_t result_slot);

    // Scalar compile/eval
    int      cm_compile(cm_program* p);
    int      cm_compile_ex(cm_program* p, const cm_jit_options* opts);
    double   cm_eval(const cm_program* p, const double* vars);

    // Batch compile/eval (SoA). inputs[j] points to array of length n.
    int      cm_compile_batch(cm_program* p);
    int      cm_compile_batch_ex(cm_program* p, const cm_jit_options* opts);
    void     cm_eval_batch(const cm_program* p,
                           const double* const* inputs, size_t n, double* out);

    int      cm_prog_is_jitted(const cm_program* p);
    int      cm_prog_is_jitted_batch(const cm_program* p);
    int      cm_jit_globally_supported(void);

#ifdef __cplusplus
}
#endif

#endif // CMATH_H