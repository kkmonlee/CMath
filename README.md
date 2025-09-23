# CMath
CMath is a tiny recursive‑descent parser and mathematical evaluation engine with two execution modes:
- A zero‑dependency ANSI C interpreter.
- An optional LLVM‑based JIT that compiles expressions to optimized machine code, with scalar and batch (SoA) kernels.

Use it to evaluate mathematical expressions at runtime and bind variables without generating code by hand. The interpreter works anywhere a C compiler runs. The JIT is optional and currently targets Apple Silicon (aarch64) first.

## Features
- Interpreter
    - ANSI C, single C file + header, no external dependencies
    - Standard operator precedence and common math functions
    - Thread‑safe (assuming your allocator is)
- Optional LLVM JIT (accelerator)
    - Scalar and batch (SoA) kernels
    - Front‑end IR + optimizer (constant folding, CSE, DCE, peepholes)
    - FMA fusion, sqrt(x*x) -> fabs(x) rewrite, integer power lowering
    - Vectorization‑friendly loops with LLVM loop metadata
    - Alias/readonly/writeonly parameter attributes
    - Tunables: unroll/interleave/vector‑width hints, prefetch distance, alignment assumptions
    - Optional nontemporal stores for streaming outputs
    - Opaque‑pointer ready (LLVM 17+)
- Deterministic numerics: JIT, interpreter, and native agree within FP roundoff; see benchmark accuracy checks.

## Example (interpreter)
    #include "cmath.h"
    printf("%f\n", cm_interp("5*5", 0)); /* Prints 25 */

## Interpreter API
CMath defines these functions (no LLVM required):

    double  cm_interp(const char *expression, int *error);
    cm_expr* cm_compile(const char *expression, const cm_variable *variables, int var_count, int *error);
    double  cm_eval(const cm_expr *expr);
    void     cm_free(cm_expr *expr);

- cm_interp: parse + evaluate in one call (returns NaN on parse error; optionally sets error position).
- cm_compile/cm_eval/cm_free: compile once, evaluate many times with current variable bindings.

Quick example with variables:

    double x=3, y=4;
    cm_variable vars[] = {{"x",&x}, {"y",&y}};
    int err = 0;
    cm_expr* e = cm_compile("sqrt(x^2+y^2)", vars, 2, &err);
    if (e) {
      printf("%f\n", cm_eval(e));  // 5.0
      cm_free(e);
    }

## Optional LLVM JIT API (accelerator)
Include `cmath_jit_llvm.h` when building with LLVM available:

    // Capability
    int cm_llvm_jit_supported(void);

    // Scalar JIT
    typedef double (*cm_jit_fn)(const double* vars);
    int cm_llvm_jit_compile(const cm_instr* code, size_t n_insts,
                            size_t num_vars, size_t num_slots, uint32_t result_slot,
                            cm_jit_fn* out_fn, void** out_state);

    // Batch JIT (SoA)
    typedef void (*cm_jit_fn_batch)(const double* const* inputs, size_t n, double* out);
    int cm_llvm_jit_compile_batch(const cm_instr* code, size_t n_insts,
                                  size_t num_vars, size_t num_slots, uint32_t result_slot,
                                  cm_jit_fn_batch* out_fn, void** out_state);

    // Tunables (optional)
    typedef struct cm_jit_options {
      int opt_level;               // 0..3 (default 3)
      int enable_const_fold;       // default 1
      int enable_cse;              // default 1
      int enable_dce;              // default 1
      int enable_auto_fma;         // default 1
      int powi_limit;              // default 8
      int vec_width_hint;          // 0=auto (default 0)
      int interleave_hint;         // default 2
      int unroll_hint;             // default 4
      int alignment;               // assumed input/out alignment bytes (default 16)
      int prefetch_distance;       // 0 disables (default 64)
      int block_size;              // strip mine size (default 0=off)
      int assume_noalias;          // add noalias/readonly/writeonly (default 1)
      int nontemporal_store;       // mark out[i] as nontemporal (default 0)
    } cm_jit_options;

    int cm_llvm_jit_compile_ex(..., const cm_jit_options* opts, ...);
    int cm_llvm_jit_compile_batch_ex(..., const cm_jit_options* opts, ...);

    // Release JIT state
    void cm_llvm_jit_release(void* state);

IR opcodes include `CM_OP_CONST, VAR, ADD, SUB, MUL, DIV, NEG, SQRT, ADD_K, MUL_K, RECIP, POWI, FMA, ABS`.

Minimal batch usage:

    // Build IR for: y = 0.5 * (fma(u, v, w) + fabs(w))
    cm_program* p = build_program_somehow();   // your IR builder
    // JIT
    cm_jit_fn_batch fn; void* st = NULL;
    cm_llvm_jit_compile_batch(p->code, p->n_insts, p->num_vars, p->num_slots, p->result_slot, &fn, &st);
    // Execute on SoA inputs
    const double* inputs[] = {U, V, W}; // SoA arrays for vars 0,1,2
    fn(inputs, N, OUT);
    cm_llvm_jit_release(st);

## Building
- Interpreter only: just add `cmath.c`/`cmath.h` to your project. No dependencies.
- With JIT:
    - Requires LLVM (tested with LLVM 17/18 on Apple Silicon).
    - CMake tips:
        - `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
        - Optionally point compilers at your LLVM toolchain:
            - `-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang`
            - `-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`

## What’s new in this version (techniques used)
Front‑end and algebra:
- IR with simple opcodes and immediate forms (`ADD_K`, `MUL_K`, `RECIP`, `POWI`, `FMA`, `ABS`).
- Constant folding, common subexpression elimination, dead code elimination.
- Peepholes:
    - FMA fusion from add/sub‑of‑mul shapes to a single `llvm.fma`.
    - sqrt(x*x)→fabs(x) rewrite (mathematically exact, cheaper than sqrt).
    - Integer power lowering with small‑exponent specialization and exponent clamp.

JIT codegen:
- ORC `LLJIT` backend, per‑module O3 pipeline via `PassBuilder`.
- Fast‑math flags on FP ops and intrinsics; explicit `llvm.fma`, `llvm.sqrt`, `llvm.fabs`.
- Opaque‑pointer migration: `PointerType::getUnqual(Context)` + typed GEP/loads/stores.

Batch kernel (SoA):
- Single tight loop with:
    - Loop metadata hints for vectorize/unroll/interleave.
    - Optional prefetch of future elements.
    - `noalias`/`readonly`/`writeonly` parameter attributes when safe.
    - Alignment assumptions exposed to LLVM.
    - Optional nontemporal stores for streaming `out[i]`.

Portability and tooling:
- Works with recent clang/LLVM (17–18) on Apple Silicon; interpreter remains portable C.

## Speed
Below is the main SoA throughput test that compares batch JIT vs a native batch loop.
- Platform: Apple Silicon (aarch64), N=1,048,576 elements per run.
- Accuracy check (10,000 samples): max abs error = `0.000e+00`; mean abs error = `0.000e+00`.

Throughput (SoA), representative run:

    JIT batch (fused):    1257.29 M eval/s   (time ~0.001 s)
    Native batch (fused):  882.64 M eval/s   (time ~0.001 s)
    Ratio (JIT/native):       1.42×  (JIT faster)

For the best JIT performance on large streaming kernels, I have found that
- In the native reference, it is better to prefer `__builtin_fma`, `__builtin_fabs` (or enable `-ffp-contract=fast`) to avoid libm calls that inhibit vectorization.
- Align inputs/outputs (e.g., 64 bytes) and set `cm_jit_options.alignment` accordingly.
- If `out[]` is not reread soon, set `opts.nontemporal_store = 1`.
- Consider moderate `unroll_hint`/`interleave_hint`, and `prefetch_distance` for regular access patterns.

## Grammar
    <list>      =    <expr> {"," <expr>}
    <expr>      =    <term> {("+" | "-") <term>}
    <term>      =    <factor> {("*" | "/" | "%") <factor>}
    <factor>    =    <power> {"^" <power>}
    <power>     =    {("-" | "+")} <base>
    <base>      =    <constant> | <variable> | <function-0> {"(" ")"} | <function-1> <power> | <function-2> "(" <expr> "," <expr> ")" | "(" <list> ")"

- Whitespace between tokens is ignored.
- Variables: lowercase a–z.
- Constants: integers, decimals, scientific notation (e.g., 1e3).
- Exponentiation associates left‑to‑right.

## Functions supported
- Operators: +, −, *, /, %, ^ (see precedence above)
- Math functions:
  abs (fabs), acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, ln (log), log (log10), pow, sin, sinh, sqrt, tan, tanh
- Constants: `pi`, `e`

## License
GNU GPL v3

