#ifndef CMATH_JIT_LLVM_H
#define CMATH_JIT_LLVM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*cm_jit_fn)(const double* vars);
typedef void   (*cm_jit_fn_batch)(const double* const* inputs, size_t n, double* out);

// Opcodes
typedef enum cm_opcode {
  CM_OP_CONST = 0,
  CM_OP_VAR   = 1,
  CM_OP_ADD   = 2,
  CM_OP_SUB   = 3,
  CM_OP_MUL   = 4,
  CM_OP_DIV   = 5,
  CM_OP_NEG   = 6,
  CM_OP_SQRT  = 7,
  CM_OP_ADD_K = 8,
  CM_OP_MUL_K = 9,
  CM_OP_RECIP = 10,
  CM_OP_POWI  = 11,
  CM_OP_FMA   = 12,
  CM_OP_ABS   = 13
} cm_opcode;

typedef struct cm_instr {
  uint32_t op;
  uint32_t dst;
  uint32_t a;
  uint32_t b;
  uint32_t c;
  int32_t  aux;
  double   imm;
} cm_instr;

typedef struct cm_jit_options {
  int opt_level;            // 0..3 (default 3)
  int enable_const_fold;    // default 1
  int enable_cse;           // default 1
  int enable_dce;           // default 1
  int enable_auto_fma;      // default 1
  int powi_limit;           // default 8

  // Batch kernel tuning
  int vec_width_hint;       // 0=auto, else 2/4/8... (default 0)
  int interleave_hint;      // default 2
  int unroll_hint;          // default 4
  int alignment;            // assumed alignment (bytes) for inputs/out (default 16)
  int prefetch_distance;    // 0 disables (default 64)
  int block_size;           // strip-mined block size (0 disables, default 0)
  int assume_noalias;       // add noalias/readonly/writeonly (default 1)
  int nontemporal_store;    // mark out[i] store as nontemporal (default 0)
} cm_jit_options;

// Capability query
int cm_llvm_jit_supported(void);

// Scalar compile
int cm_llvm_jit_compile(const cm_instr* code,
                        size_t n_insts,
                        size_t num_vars,
                        size_t num_slots,
                        uint32_t result_slot,
                        cm_jit_fn* out_fn,
                        void** out_state);

int cm_llvm_jit_compile_ex(const cm_instr* code,
                           size_t n_insts,
                           size_t num_vars,
                           size_t num_slots,
                           uint32_t result_slot,
                           const cm_jit_options* opts,
                           cm_jit_fn* out_fn,
                           void** out_state);

// Batch compile (SoA)
int cm_llvm_jit_compile_batch(const cm_instr* code,
                              size_t n_insts,
                              size_t num_vars,
                              size_t num_slots,
                              uint32_t result_slot,
                              cm_jit_fn_batch* out_fn,
                              void** out_state);

int cm_llvm_jit_compile_batch_ex(const cm_instr* code,
                                 size_t n_insts,
                                 size_t num_vars,
                                 size_t num_slots,
                                 uint32_t result_slot,
                                 const cm_jit_options* opts,
                                 cm_jit_fn_batch* out_fn,
                                 void** out_state);

// Release
void cm_llvm_jit_release(void* state);

#ifdef __cplusplus
}
#endif

#endif // CMATH_JIT_LLVM_H