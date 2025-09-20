# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CMath — Project Summary (updated with advanced techniques)

CMath is a lightweight recursive descent parser and mathematical evaluation engine written in ANSI C. It provides runtime evaluation of mathematical expressions with support for variables, standard mathematical functions, and thread-safe operation.

## Core Architecture

### Single File Library Structure

* **cmath.h**: Main header defining the API and data structures
* **cmath.c**: Complete implementation of the parser and evaluator
* **example.c**: Simple usage demonstration
* **benchmark.c**: Performance comparison tool

### Key Data Structures

* `cm_expr`: expression tree node with optimization metadata (bytecode, pattern, flags)
* `cm_variable`: variable binding structure for runtime variable resolution
* `cm_expr_pool`: thread-safe memory pool with mutex protection and compaction
* `cm_bytecode`: stack-based virtual machine instructions for fast evaluation
* `cm_pattern`: pattern analysis for specialized evaluation paths

### Threading Model

thread-safety through:

* global expression pool with pthread mutex protection (`globalPool`)
* memory compaction and reuse strategy to avoid fragmentation
* pool-based allocation system in `cmath.c:35-42`

## Build System

### CMake Build

```bash
mkdir -p build
cd build
cmake ..
make
```

**Note**: All compilation errors have been resolved. The library now builds successfully with advanced optimizations.

### Direct Compilation

compilation with full optimizations for Apple Silicon:

```bash
gcc -o example example.c cmath.c -std=c99 -O3 -march=native -ffast-math -lpthread -lm
gcc -o benchmark benchmark.c cmath.c -std=c99 -O3 -march=native -ffast-math -lpthread -lm
```

## API Usage Patterns

### Primary Functions

1. `cm_interp()`: One-shot expression evaluation
2. `cm_compile()`: Parse expression with variable bindings
3. `cm_eval()`: Evaluate pre-compiled expression
4. `cm_free()`: Release expression memory

### Common Usage Pattern

```c
double x, y;
cm_variable vars[] = {{"x", &x}, {"y", &y}};
cm_expr *expr = cm_compile("sqrt(x^2+y^2)", vars, 2, &err);

x = 3; y = 4;
double result1 = cm_eval(expr, NULL);
x = 5; y = 12;
double result2 = cm_eval(expr, NULL);

cm_free(expr);
```

## Parser Implementation

### Grammar Structure

* recursive descent parser implementing standard operator precedence
* left-to-right exponentiation evaluation
* standard C math functions (sin, cos, sqrt, log, etc.)
* constants: pi, e

### Memory Management

* pool-based allocation with compaction strategy
* thread-safe node allocation and deallocation
* automatic memory cleanup through `cm_free()`
* optimization structure cleanup (bytecode, patterns, JIT code)

## Development Notes

### Advanced Optimizations

the codebase includes comprehensive performance optimizations:

* ARM NEON SIMD vectorization for Apple Silicon
* compile-time constant folding and algebraic simplification
* bytecode virtual machine for efficient evaluation
* pattern analysis and specialized evaluation paths
* cache-friendly memory layout optimization

### Performance Characteristics

* designed for high-performance repeated evaluation
* simple expressions: only 5-6% slower than native C (270M evals/sec)
* complex expressions: significant improvements over baseline interpreter
* memory-efficient through expression tree reuse and optimization caching
* Apple Silicon optimized with ARM NEON SIMD instructions

---

## Advanced Techniques (borrowed from Intel MKL, AMD LibM, SLEEF, Yeppp!, OpenLibm, CRlibm, and Vectorized BLAS/LAPACK libraries)

Below are advanced methods these libraries use that you can incorporate into CMath. Each item includes **what it is**, **why it helps**, and **how to implement it in CMath** (parser / bytecode / VM / pool / JIT / evaluation).

> Implement these progressively — start with vectorization, range reduction & polynomial approximations, then add CPU dispatch, JIT, and correctness/performance trade-offs.

### 1. SIMD / Vectorized Evaluation (SIMD intrinsics & batch API)

**What / Why:** Evaluate many values in parallel (SIMD) instead of scalar per-value calls. Libraries like MKL/VML, SLEEF, Yeppp! expose vectorized math that uses SSE/AVX/AVX-512/NEON/SVE for dramatic throughput wins on arrays/batched workloads.
**How to add to CMath:**

* Add a batched-eval API: `cm_eval_vec(cm_expr *e, double *out, const double **vars, size_t n)` that evaluates the same expression across `n` inputs.
* Implement SIMD kernels for common functions: `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`. Start with NEON (Apple Silicon) and x86 AVX2/AVX-512 intrinsics.
* Use vector-friendly loop structure and memory alignment (aligned pools, `posix_memalign`) and process in blocks of vector width (4/8/16).
* Implement a vectorized bytecode path (special bytecode opcodes that operate on SIMD registers rather than scalars).
* Add fallback scalar path for remainders.

### 2. Fast Polynomial / Rational Approximations (minimax / Chebyshev / Estrin)

**What / Why:** Transcendental functions are implemented via small-degree polynomials or rational functions tuned to minimize max error on a reduced interval. Estrin’s scheme helps evaluate polynomials with fewer dependencies (better for SIMD).
**How to add:**

* Replace naive `math.h` calls inside your VM with polynomial approximations for target ranges.
* Use precomputed coefficients for intervals. Implement an **Estrin evaluation** routine for SIMD-friendly polynomial evaluation.
* Provide different approximation degrees selectable at compile/runtime (accuracy vs speed).
* Store coefficient tables in the expression structure or globally in the pool (cache-friendly).

### 3. Range Reduction and Payne–Hanek / Cody–Waite methods (for trig and log)

**What / Why:** Reducing input to a small core interval (e.g., reduce angle for sin/cos to $-π/4, π/4$) avoids huge polynomials and keeps approximations accurate. Payne–Hanek is very accurate for large inputs.
**How to add:**

* Implement robust range reduction routines for `sin`, `cos`, `tan` (Payne–Hanek when needed). Implement a fast path for small inputs (abs(x) < threshold) that avoids full reduction.
* Make range reduction vectorized where possible.
* Add small-angle approximations for x near 0 (use Taylor with rounding-safe coefficients).

### 4. Table-driven Approximations + Hybrid LUT+Poly (lookup + tiny correction)

**What / Why:** Use a small lookup table for coarse approximation and apply a low-degree polynomial to correct residual — gives faster and more accurate results than a single polynomial for wide ranges.
**How to add:**

* Build small uniform lookup tables for functions like `exp`, `sin`, and `log`. Use table index derived from input bits (mantissa) after range reduction.
* Interpolate with a low-degree polynomial (or rational) on the error term.
* Store tables in read-only contiguous arrays (cache line optimized) in `cmath.c` (or in the pool when JITing).

### 5. Fused Multiply-Add (FMA) and Hardware Instructions

**What / Why:** FMA reduces rounding and instruction count for polynomial evaluation. Use CPU FMA intrinsics to speed inner loops.
**How to add:**

* Use `-mfma` or compiler intrinsics (`__builtin_fma`, `_mm256_fmadd_pd`, etc.) in polynomial evaluation and Newton iterations. Provide compile-time guards and runtime dispatch if target CPU lacks FMA.
* Use FMA inside Estrin/Horner implementations.

### 6. Newton-Raphson & Iterative Refinement for Inverses / sqrt / division

**What / Why:** Use fast hardware approximate instructions (e.g., `rsqrt`/`vrsqrte`) then refine with 1–2 Newton iterations to reach double precision rapidly.
**How to add:**

* Implement `fast_rsqrt` and `fast_recip` using hardware approximation + 1–2 Newton steps.
* Use these for `1/x` and `sqrt` computations in vectorized paths.

### 7. Hardware-targeted code via CPU feature dispatch & function multi-versioning

**What / Why:** Compile several versions (AVX2, AVX-512, NEON, scalar) and dispatch to the best implementation at runtime. Yeppp!/SLEEF/MKL do this extensively.
**How to add:**

* Build multiple compiled code paths and select at runtime using CPUID on x86 or `sysctl`/`mach` on macOS. Use weak-linking or function pointers set at init.
* Keep a clean abstraction: `cm_fn_sin = &cm_sin_avx2` etc.
* Add compile-time `#ifdef` and build targets for major ISAs.

### 8. Vectorized Special-case / Branchless Handling & Bit-twiddling

**What / Why:** Branches hurt SIMD. Implement branchless checks (masking) for NaN, Inf, sign bits, and special cases using bitwise ops and masked moves.
**How to add:**

* Use integer masks after bit-casting floats to ints to detect special cases without branching.
* Use masked blend instructions in AVX/NEON for selects.
* Fallback to scalar math for irregular cases only.

### 9. Correctness Modes & Error/Accuracy Tiers (trade speed vs rounding)

**What / Why:** Libraries expose accuracy modes (fast/approximate vs accurate/correctly-rounded). CRlibm focuses on correct rounding. Give users modes.
**How to add:**

* Offer API flags: `CM_FAST`, `CM_BALANCED`, `CM_CORRECT_ROUNDED`.
* Implement faster approximations for `CM_FAST`, and full Payne–Hanek + high-precision corrections for `CM_CORRECT_ROUNDED`.
* Document numerical guarantees per mode.

### 10. Vectorized Transcendental Algorithms (SLEEF-style)

**What / Why:** SLEEF implements vector math with small polynomials, range reduction, and careful handling of edge cases — optimized for each SIMD ISA.
**How to add:**

* Study and implement SLEEF-like kernels (or integrate SLEEF as an optional backend).
* For each function, create a vector kernel and a scalar fallback.
* Integrate these kernels into the bytecode VM’s vectorized opcodes.

### 11. Lookup & Precompute at Compile/Compile-time (constant folding + precompute)

**What / Why:** Precompute parts of expressions at compile/parse time (constant folding) so evaluations avoid redundant computation.
**How to add:**

* Expand compile-time constant folding: evaluate any pure subexpression at `cm_compile()` and replace with constant node.
* Precompute function approximation data per expression where possible and store in `cm_expr` (e.g., small LUTs for a specific expression pattern).

### 12. Pattern Analysis → Specialized Code Paths (like VML’s specialized kernels)

**What / Why:** Detect common patterns (vector norms, dot products, polynomial expressions) and emit a specialized, faster path.
**How to add:**

* Extend `cm_pattern` to detect: `sqrt(x*x + y*y)`, `sum of products`, repeated subexpressions, or simple linear combinations.
* For patterns, generate specialized bytecode or JITed native code (see 13) that uses vector instructions and fused ops.

### 13. JIT / Native Code Emission for Hot Expressions

**What / Why:** Emit native machine code for a compiled expression. This removes interpreter overhead and allows expression-specific optimization and inlining. Many fast systems JIT expensive functions.
**How to add:**

* Add optional JIT backend that emits x86-64/ARM64 code (use a small JIT library or a simple codegen layer).
* Use pattern analysis to generate code that uses registers, FMA, and vector loads/stores.
* Cache JIT code in `cm_expr` and guard with a max-size / memory policy in the pool.
* Fallback to bytecode VM if JIT unavailable.

### 14. Bytecode-to-SIMD Translation & VM Optimizations

**What / Why:** Keep the VM but make its hot path SIMD-aware; reduce per-op overhead, dispatch cost.
**How to add:**

* Introduce fused bytecode ops that perform whole subexpressions in one call (e.g., `OP_VEC_EXP_ADD`).
* Use threaded dispatch, computed goto (`labels as values`) for faster interpreter loops.
* Combine stack operations and eliminate unnecessary pushes/pops during compilation (peephole optimization).

### 15. Cache-friendly Data Layout & Memory Alignment

**What / Why:** Contiguous, aligned arrays reduce cache misses and improve vector load efficiency.
**How to add:**

* Use SoA (structure of arrays) layout for batched variable inputs where appropriate.
* Allocate bytecode/coeff tables in a pool region aligned to 64 bytes.
* Use memory compaction to keep frequently-used data dense in memory.

### 16. Loop Unrolling & Software Pipelining in Hot Loops

**What / Why:** Reduce loop overhead and enable the CPU to schedule independent operations.
**How to add:**

* Unroll vector evaluation loops (process multiple vector blocks per iteration).
* Use compiler pragmas where helpful (`#pragma GCC unroll`), but prefer manual unrolling for predictable control.

### 17. Use of Approximate Hardware Instructions & Iteration to Correct

**What / Why:** Many ISAs offer fast approximate instructions (e.g., `vrsqrt`, `reciprocal estimate`) — combine them with Newton steps to reach target precision much faster than full hardware division.
**How to add:**

* Implement approximate path using `vrsqrte`/`vrecpeq` + refine.
* Keep approximate-only modes for `CM_FAST`.

### 18. Multiple-Precision / Extended Precision Paths for Edge Cases (CRlibm concept)

**What / Why:** To provide correctly rounded results, compute in extended precision (quad or double-double) or use multiple-precision intermediate steps only for problematic inputs.
**How to add:**

* Implement optional double-double arithmetic or soft-float extended routines for final correction in `CM_CORRECT_ROUNDED` mode.
* Only enable on inputs that fail quick checks (rare), keeping the fast path untouched.

### 19. Batched and Threaded Evaluation

**What / Why:** For large workloads, multithread the batches (MKL/VML style) and minimize synchronization.
**How to add:**

* Add `cm_eval_vec_mt` which partitions the input across threads (thread pool).
* Use per-thread scratch buffers from `cm_expr_pool` to avoid locking on hot paths.
* Keep thread-safety by making read-only data immutable after compile.

### 20. Runtime Accuracy/Performance Tuning & Auto-Tuning

**What / Why:** Auto-tune polynomial degree, table sizes, and vector block sizes for the specific CPU and workload. Libraries ship tuned parameter sets.
**How to add:**

* Add a small runtime tuner invoked first time `cm` runs on a CPU that benchmarks candidate kernel variants and stores decision in `globalPool` or a config file.
* Or allow compile-time flags for predetermined targets.

### 21. Integration/Optional Backend Strategy

**What / Why:** Instead of reimplementing everything, allow CMath to plug in fast backends (SLEEF, libm replacements, MKL) at build or runtime.
**How to add:**

* Provide a backend interface (`cm_backend_ops`) with function pointers for `sin`, `exp`, `log`, etc.
* Implement adapters for SLEEF, AMD LibM (if available), or vendor MKL VML. Allow dynamic linking or compile-time selection.
* Fall back to built-in scalar/vector implementations when backends are absent.

---

## Mapping techniques to CMath components (actionable checklist)

**Parser / `cm_compile()`**

* [ ] Add pattern detection (norms, dot-products, repeated subexpressions).
* [ ] Constant-fold all pure subexpressions.
* [ ] Precompute per-expression LUTs or coefficients where applicable.

**Bytecode / VM**

* [ ] Add fused bytecode ops for common patterns.
* [ ] Add vectorized opcodes (operate on `doublex4/doublex8`).
* [ ] Implement computed-goto dispatch for speed.
* [ ] Implement interpreter-level peephole and CSE optimizations.

**Evaluation / Runtime**

* [ ] Add batched API `cm_eval_vec` and multithreaded `cm_eval_vec_mt`.
* [ ] Implement vector kernels for math functions (NEON / AVX2 / AVX-512).
* [ ] Use Estrin/Horner with FMA in polynomial evaluation.
* [ ] Implement robust range reduction (Payne–Hanek) and table-driven approaches.
* [ ] Add accuracy modes (FAST / BALANCED / CORRECT\_ROUNDED).
* [ ] Add runtime CPU dispatch and multi-versioning.
* [ ] Add branchless special-case handling and fallback scalar path.

**Memory / Pools**

* [ ] Align coefficient tables and vector buffers to 64 bytes.
* [ ] Provide per-thread scratch buffers in the pool to avoid contention.
* [ ] Store JIT code and LUTs in compact, cache-friendly regions.

**Optional JIT**

* [ ] Implement a JIT codegen path for hot expressions, emitting FMA, SIMD loads/stores, and expression-specific constants.
* [ ] Cache and evict JITed code carefully; respect `cm_free()` semantics.

**Backends**

* [ ] Provide a pluggable backend interface and adapters for SLEEF / AMD LibM / MKL / OpenLibm.

---

## Practical implementation notes and priorities

1. **Priority 0 (biggest win, moderate work):** SIMD vectorized evaluation for batched inputs + polynomial approximations (Estrin + FMA) + range reduction.
2. **Priority 1:** CPU feature dispatch, lookup tables, branchless special-case handling, and runtime accuracy modes.
3. **Priority 2 (more work, high payoff on hot paths):** JIT native emitter for hot expressions, multi-precision corrections (CRlibm-like) for correctly-rounded mode.
4. **Priority 3:** Auto-tuning, exhaustive CPU micro-optimizations, and vendor backend integration.

---

## Example: How the `sqrt(x^2 + y^2)` path could be optimized

1. At `cm_compile()` detect `sqrt(x^2+y^2)` pattern → mark pattern as `PAT_NORM2`.
2. On `cm_eval_vec()` use a specialized vector kernel:

   * load x,y into SIMD regs (SoA or interleave),
   * compute `mul` + `add` with FMA if available,
   * use `vrsqrt` + Newton refinement for reciprocal square root or call vector `sqrt` instruction,
   * handle special cases (Inf/NaN) with masked blends only.
3. If `n` is small, use scalar fast path; if `n` large, partition and run multi-threaded with per-thread buffers.

---

## Backwards compatibility and API additions

* Add `cm_eval_mode_t` flags and new APIs:

```c
typedef enum { CM_MODE_FAST=1, CM_MODE_BALANCED=2, CM_MODE_CORRECT=4 } cm_eval_mode_t;
double cm_eval(cm_expr *e, const cm_eval_context *ctx); // existing
void cm_eval_vec(cm_expr *e, double *out, const double **vars, size_t n, cm_eval_mode_t mode);
void cm_set_backend(cm_backend_ops *ops); // optional pluggable backend
```

---

## Summary

If you implement the above techniques incrementally you will capture the core strategies that those high-performance libraries use:

* **Vectorization** (SIMD kernels + batched API),
* **Range reduction + table-driven polynomial approximations** (Estrin, minimax),
* **Hardware features** (FMA, rsqrt, intrinsics) and **runtime dispatch**,
* **Specialized patterns and JIT** for hot expressions,
* **Branchless handling** and **accuracy-mode selection**.

Start by adding a batched evaluation API and implementing vectorized polynomial kernels for the most-used functions (`exp`, `log`, `sin`, `cos`, `sqrt`), then add runtime CPU dispatch and pattern-specialized kernels. After that, consider a JIT backend and a pluggable vendor backend interface to leverage MKL/SLEEF if available.

Below are:

* **A. Concrete C code skeletons**: vector primitives, a vector-capable bytecode layout and interpreter sketch, plus *example* vectorized `exp` and `sin` kernels — **AVX2** (x86\_64) and **NEON** (aarch64/Apple Silicon) variants where feasible. These are **practical skeletons** (complete enough to compile with the right flags and to be extended), annotated heavily with TODOs and notes where production-grade implementations require extra care (table coefficients, Payne–Hanek, correct rounding, etc.).

* **B. A prioritized, elaborate roadmap** your generative AI can follow: step-by-step tasks, which C modules/functions to add, expected lines of code (LOC) *estimates* per task, testing & benchmarking notes, and the function-level breakdown so it can implement each piece reliably.

I focused on **clarity, correctness of structure, and immediate implementability**, while marking where you should later plug in high-quality coefficients or vendor backends (SLEEF, MKL, AMD LibM) for production accuracy.

---

# A — Vectorized bytecode + kernels (skeletons)

> ⚠️ Important notes before you paste & compile:
>
> * Compile AVX2 code with `-mavx2 -mfma -O3` on x86-64. For Apple Silicon (aarch64) compile with `-O3` (NEON is available by default on arm64).
> * These are **skeletons** for correctness/performance patterns. For production you must (a) replace polynomial coefficients with tuned minimax/Estrin tables, (b) add robust range reduction (Payne–Hanek) for trig, (c) add branchless special-case handling for NaN/Inf, and (d) validate accuracy across edge cases.
> * I include both AVX2 and NEON code paths guarded by `#if` checks and a simple runtime dispatch.

---

## 1) Small header / vector abstraction (paste into cmath or separate cm\_vector.h)

```c
// cm_vector.h  -- small vector abstraction for AVX2 and AArch64 NEON
#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define CM_HAVE_AVX2 1
#elif defined(__aarch64__)
  #include <arm_neon.h>
  #define CM_HAVE_NEON 1
#else
  #define CM_HAVE_SCALAR 1
#endif

// Vector width in doubles
#if defined(CM_HAVE_AVX2)
  #define CM_VW 4
  typedef __m256d cm_vd;
#elif defined(CM_HAVE_NEON)
  #define CM_VW 2
  typedef float64x2_t cm_vd;
#else
  #define CM_VW 1
  typedef double cm_vd;
#endif

// Vector helpers (set1/load/store/add/mul/fma)
// For readability we wrap intrinsics. Keep them inline for performance.

static inline cm_vd vec_set1_pd(double x) {
#if defined(CM_HAVE_AVX2)
  return _mm256_set1_pd(x);
#elif defined(CM_HAVE_NEON)
  return vdupq_n_f64(x);
#else
  return x;
#endif
}

static inline cm_vd vec_load_pd(const double *p) {
#if defined(CM_HAVE_AVX2)
  return _mm256_loadu_pd(p);
#elif defined(CM_HAVE_NEON)
  return vld1q_f64(p);
#else
  return *p;
#endif
}

static inline void vec_store_pd(double *p, cm_vd v) {
#if defined(CM_HAVE_AVX2)
  _mm256_storeu_pd(p, v);
#elif defined(CM_HAVE_NEON)
  vst1q_f64(p, v);
#else
  *p = v;
#endif
}

static inline cm_vd vec_add_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_add_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vaddq_f64(a,b);
#else
  return a + b;
#endif
}

static inline cm_vd vec_sub_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_sub_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vsubq_f64(a,b);
#else
  return a - b;
#endif
}

static inline cm_vd vec_mul_pd(cm_vd a, cm_vd b) {
#if defined(CM_HAVE_AVX2)
  return _mm256_mul_pd(a,b);
#elif defined(CM_HAVE_NEON)
  return vmulq_f64(a,b);
#else
  return a * b;
#endif
}

static inline cm_vd vec_fma_pd(cm_vd a, cm_vd b, cm_vd c) {
#if defined(CM_HAVE_AVX2)
  return _mm256_fmadd_pd(a,b,c);  // requires -mfma
#elif defined(CM_HAVE_NEON)
  // vfmaq_f64 requires armv8.2-a with FP16/FMA; emulate if not available:
  #if defined(__ARM_FEATURE_FMA)
    return vfmaq_f64(c, a, b);
  #else
    return vaddq_f64(vmulq_f64(a,b), c);
  #endif
#else
  return a*b + c;
#endif
}
```

---

## 2) Vector polynomial evaluation (Estrin/Horner building block)

This small function evaluates a polynomial `p(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n`. For simplicity I provide a **Horner**-style evaluator using FMA which is safe and easy; you can switch to Estrin for wider SIMD pipelines later.

```c
// evaluate polynomial using Horner in vector form
// coeffs: array coeff[0..deg] (coeff[0] = c0)
// deg: degree (>=0)
static inline cm_vd vec_poly_horner(const double *coeffs, int deg, cm_vd x) {
    // compute c_deg * x + c_(deg-1) ... using FMA
    cm_vd acc = vec_set1_pd(coeffs[deg]);
    for (int i = deg-1; i >= 0; --i) {
        acc = vec_fma_pd(acc, x, vec_set1_pd(coeffs[i]));
    }
    return acc;
}
```

**TODO:** implement `vec_poly_estrin()` for better ILP on large degrees.

---

## 3) Vectorized `exp` kernel (AVX2 and NEON skeleton)

The following code uses **range reduction** `x = k*ln2 + r`, evaluates `exp(r)` with a polynomial, then multiplies by `2^k` via exponent-bit construction (fast) in the AVX2 path. The NEON path uses a simpler scalar `ldexp` per lane (because building bit patterns in NEON is more verbose here). Replace `coeffs` with minimax coefficients for real performance/accuracy.

```c
// fast vectorized exp skeleton (not production-correct for all edge cases)
// Use compile flags: -mavx2 -mfma on x86; aarch64 NEON for arm.
#include <math.h>
#include <string.h>

static const double EXP_POLY_COEFFS[] = {
    // placeholder coefficients: MUST be replaced with tuned minimax
    1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0  // up to x^5 Taylor
};
static const int EXP_POLY_DEG = 5;

static inline void vec_exp_avx2(const double *in, double *out) {
#if defined(CM_HAVE_AVX2)
    const __m256d ln2 = _mm256_set1_pd(0.693147180559945309417232121458176568);
    const __m256d inv_ln2 = _mm256_set1_pd(1.442695040888963407359924681001892137);
    const double EXP_LO = -709.782712893384; // approx
    const double EXP_HI = 709.782712893384;

    size_t i = 0;
    for (; i + 4 <= SIZE_MAX; i += 4) {
        __m256d x = _mm256_loadu_pd(in + i);

        // clamp to avoid overflow
        __m256d x_clamped = _mm256_min_pd(_mm256_max_pd(x, _mm256_set1_pd(EXP_LO)), _mm256_set1_pd(EXP_HI));

        // k = floor(x / ln2)
        __m256d x_scaled = _mm256_mul_pd(x_clamped, inv_ln2);
        __m256d k_real = _mm256_floor_pd(x_scaled);  // requires -mavx
        __m128i k32 = _mm256_cvttpd_epi32(k_real);  // convert 4 doubles -> 4 int32 in 128-bit lane
        __m256i k64 = _mm256_cvtepi32_epi64(k32);   // expand to 4x int64
        // r = x - k*ln2
        __m256d k_d = _mm256_cvtepi64_pd(k64);
        __m256d r = _mm256_sub_pd(x_clamped, _mm256_mul_pd(k_d, ln2));

        // compute exp(r) via polynomial (Horner with FMA)
        // create coeff vectors - for demo we use EXP_POLY_COEFFS (small degree)
        __m256d acc = _mm256_set1_pd(EXP_POLY_COEFFS[EXP_POLY_DEG]);
        for (int j = EXP_POLY_DEG - 1; j >= 0; --j) {
            acc = _mm256_fmadd_pd(acc, r, _mm256_set1_pd(EXP_POLY_COEFFS[j]));
        }

        // compute 2^k by building exponent bits: bits = (k + 1023) << 52
        __m256i biased = _mm256_add_epi64(k64, _mm256_set1_epi64x(1023));
        __m256i bits = _mm256_slli_epi64(biased, 52);
        __m256d pow2 = _mm256_castsi256_pd(bits);

        __m256d result = _mm256_mul_pd(acc, pow2);
        _mm256_storeu_pd(out + i, result);
    }
    // remainder scalar fallback (if input length not multiple of 4) should be handled outside
#endif
}
```

NEON variant (aarch64) — simpler skeleton: evaluate polynomial vectorized, then compute `ldexp` per-lane for `2^k`. On aarch64 you can implement integer bit-building with `vreinterpretq_f64_s64` and shifts, but it's more verbose; for clarity we do lane-scalar `ldexp` (cheap for VW=2).

```c
static inline void vec_exp_neon(const double *in, double *out, size_t n) {
#if defined(CM_HAVE_NEON)
    const double ln2 = 0.693147180559945309417232121458176568;
    const double inv_ln2 = 1.442695040888963407359924681001892137;
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t x = vld1q_f64(in + i);
        // clamp - lanes
        // ... (similar lane-wise clamping)
        // For demonstration compute lanewise:
        double xi0 = vgetq_lane_f64(x, 0), xi1 = vgetq_lane_f64(x,1);
        int64_t k0 = (int64_t) floor(xi0 * inv_ln2);
        int64_t k1 = (int64_t) floor(xi1 * inv_ln2);
        double r0 = xi0 - (double)k0 * ln2;
        double r1 = xi1 - (double)k1 * ln2;
        // polynomial (scalar here - but you should vectorize polynomial evaluation too)
        double acc0 = EXP_POLY_COEFFS[EXP_POLY_DEG];
        for (int j = EXP_POLY_DEG -1; j >= 0; --j) acc0 = acc0 * r0 + EXP_POLY_COEFFS[j];
        double acc1 = EXP_POLY_COEFFS[EXP_POLY_DEG];
        for (int j = EXP_POLY_DEG -1; j >= 0; --j) acc1 = acc1 * r1 + EXP_POLY_COEFFS[j];

        double res0 = ldexp(acc0, (int)k0);
        double res1 = ldexp(acc1, (int)k1);
        float64x2_t res = { res0, res1 };
        vst1q_f64(out + i, res);
    }
    // remainder lanes fallback
#endif
}
```

**Notes & TODOs**

* The AVX2 path uses bit-level exponent construction (`(k+1023)<<52`) — very fast. Ensure `k` fits in int32 or clamp to safe range beforehand (we clamp x to avoid overflow).
* Replace placeholder polynomial coefficients with *minimax* coefficients or SLEEF’s coefficients for accuracy.
* Add **branchless** special-case handling for NaN/Inf using bit-masks and masked blends.
* Add a high-accuracy fallback (double-double or Newton refinement) for `CM_MODE_CORRECT`.

---

## 4) Vectorized `sin` kernel (skeleton)

`sin` requires good range reduction (argument reduction to `[-pi/4, pi/4]`), switching to polynomial, and sign/permute logic. Below is a simplified skeleton—*replace the range reduction with Payne–Hanek for large inputs before shipping*.

```c
// vectorized sin skeleton (AVX2)
static const double PI = 3.141592653589793238462643383279502884;
static const double PI_2 = 1.57079632679489661923132169163975144;
static const double INV_PI_2 = 0.636619772367581343075535053490057448; // 2/pi

// small-degree polynomial for sin(r) around r~=0 (placeholder)
static const double SIN_COEFFS[] = {
  0.0, // c0 = 0
  1.0, // c1 = 1
  0.0, // c2
  -1.0/6.0,
  0.0,
  1.0/120.0
};
static const int SIN_DEG = 5;

static inline void vec_sin_avx2(const double *in, double *out, size_t n) {
#if defined(CM_HAVE_AVX2)
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m256d x = _mm256_loadu_pd(in + i);

    // Reduce range: y = x * 2/pi ; n = round(y)
    __m256d y = _mm256_mul_pd(x, _mm256_set1_pd(INV_PI_2));
    __m256d nreal = _mm256_round_pd(y, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    __m128i n32 = _mm256_cvttpd_epi32(nreal);
    __m256i n64 = _mm256_cvtepi32_epi64(n32);
    __m256d n_d = _mm256_cvtepi64_pd(n64);

    // r = x - n * (pi/2)
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n_d, _mm256_set1_pd(PI_2)));

    // compute sin(r) via polynomial (Horner)
    __m256d acc = _mm256_set1_pd(SIN_COEFFS[SIN_DEG]);
    for (int j = SIN_DEG -1; j >= 0; --j) {
       acc = _mm256_fmadd_pd(acc, r, _mm256_set1_pd(SIN_COEFFS[j]));
    }

    // Note: this simplistic approach neglects quadrant-based sign/permute.
    // For correct sin: determine quadrant = n & 3, and apply sign/replace by cos for some quadrants.
    // Implement branchless sign flips via masks.

    _mm256_storeu_pd(out + i, acc);
  }
#endif
}
```

**IMPORTANT:** quadrant handling and robust range reduction are essential. For production, follow full `SLEEF` approach: range-reduce to the principal region with correct high-precision constants, handle large inputs with Payne–Hanek, then apply minimax polynomial for sin/cos.

---

## 5) Vector-enabled bytecode design & interpreter sketch

Design choices:

* Keep existing scalar bytecode. Add new opcodes for vectorized versions (prefix `V_`).
* Provide `cm_eval_vec(cm_expr*, double *out, const double **vars, size_t n, mode)` which dispatches to vector kernels for batched eval.
* Use a vector **stack** with type `cm_vd stack[VSTACK_SZ]`.

Bytecode layout (example):

```c
// opcode values (example)
enum {
  OP_HALT = 0,
  OP_LOAD_CONST,
  OP_LOAD_VAR,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_SIN,    // scalar
  OP_EXP,    // scalar
  // ... existing scalar ops ...
  // vector ops
  OP_V_LOAD_CONST,
  OP_V_LOAD_VAR,
  OP_V_ADD,
  OP_V_MUL,
  OP_V_EXP,   // vectorized exp
  OP_V_SIN,   // vectorized sin
  OP_V_CALL,  // call backend
};
```

Interpreter sketch:

```c
// simplified vector VM evaluation - process BLOCK of VEC_WIDTH elements at a time
void vm_eval_vec(const uint8_t *code, const double *consts, const double **vars, double *out, size_t n) {
    size_t i = 0;
    while (i + CM_VW <= n) {
        // vector stack
        cm_vd vstack[128];
        int vsp = 0;

        const uint8_t *pc = code;
        while (1) {
            uint8_t op = *pc++;
            switch (op) {
                case OP_V_LOAD_CONST: {
                    uint32_t idx = *(uint32_t*)pc; pc += 4;
                    cm_vd v = vec_load_pd(consts + idx * CM_VW + i); // if you stored per-const array of size n
                    vstack[vsp++] = v;
                    break;
                }
                case OP_V_LOAD_VAR: {
                    uint32_t var_id = *(uint32_t*)pc; pc += 4;
                    // vars[var_id] is pointer to base array; load CM_VW lanes
                    cm_vd v = vec_load_pd(vars[var_id] + i);
                    vstack[vsp++] = v;
                    break;
                }
                case OP_V_ADD: {
                    cm_vd b = vstack[--vsp];
                    cm_vd a = vstack[--vsp];
                    vstack[vsp++] = vec_add_pd(a,b);
                    break;
                }
                case OP_V_MUL: {
                    cm_vd b = vstack[--vsp];
                    cm_vd a = vstack[--vsp];
                    vstack[vsp++] = vec_mul_pd(a,b);
                    break;
                }
                case OP_V_EXP: {
                    // pop argument vector and push result vector using our kernel
                    cm_vd a = vstack[--vsp];
                    // For demo, call a small wrapper that transforms vector register to memory, calls vec_exp_*, reload
                    double tmp_in[CM_VW], tmp_out[CM_VW];
                    vec_store_pd(tmp_in, a);
                    #if defined(CM_HAVE_AVX2)
                      vec_exp_avx2(tmp_in, tmp_out);
                    #elif defined(CM_HAVE_NEON)
                      vec_exp_neon(tmp_in, tmp_out, CM_VW);
                    #else
                      for (int k=0;k<CM_VW;++k) tmp_out[k] = exp(tmp_in[k]);
                    #endif
                    cm_vd r = vec_load_pd(tmp_out);
                    vstack[vsp++] = r;
                    break;
                }
                case OP_HALT: {
                    cm_vd res = vstack[--vsp];
                    vec_store_pd(out + i, res);
                    goto next_block;
                }
                default:
                    // handle others
                    break;
            }
        }
        next_block:
        i += CM_VW;
    }

    // handle tail (n % CM_VW) in scalar mode using existing cm_eval()
}
```

**Notes**

* For performance, avoid flip-to-memory conversions inside each opcode. The example above does so for clarity only. The real implementation should call vector kernels directly on `cm_vd` values (i.e., pass register data to `vec_exp_inplace(cm_vd x)` returning cm\_vd) — that requires writing kernel functions accepting `cm_vd` vectors (no memory roundtrip).
* Keep `cm_expr` bytecode annotated with vectorizable flags so `cm_compile()` can emit `OP_V_` ops for vectorizable subexpressions only.

---

## 6) Runtime CPU dispatch & function pointers

Simple pattern:

```c
typedef void (*cm_vec_exp_fn)(const double* in, double* out, size_t n);

static cm_vec_exp_fn global_vec_exp = NULL;

static void init_cpu_dispatch(void) {
#if defined(CM_HAVE_AVX2)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    global_vec_exp = vec_exp_avx2;
    return;
  }
#endif
#if defined(CM_HAVE_NEON)
  global_vec_exp = vec_exp_neon;
  return;
#endif
  // fallback:
  global_vec_exp = NULL;
}
```

Call `init_cpu_dispatch()` at library init (e.g., first `cm_compile()` or explicit `cm_init()`).

---

# B — Prioritized, actionable roadmap + LOC/function breakdown

Below is a practical, prioritized plan that your generative AI can implement **step-by-step**. Each task includes what to implement, how it interacts with other parts, and a rough **LOC estimate** to help schedule work (LOC = lines of C + comments + tests). I do **not** provide time estimates.

---

## Phase 0 — Prep & scaffolding (core infra)

**Goals:** Add vector abstraction, runtime dispatch, and new APIs for batched evaluation.

1. Add `cm_vector.h` and include in `cmath.c`.

   * Files: `cm_vector.h` (vector wrappers), updates to `cmath.h` to expose `cm_eval_vec`.
   * Functions: `vec_set1_pd`, `vec_load_pd`, `vec_store_pd`, `vec_add_pd`, `vec_mul_pd`, `vec_fma_pd`.
   * LOC: **\~120 LOC**

2. Add runtime dispatch module `cm_dispatch.c` / functions inside `cmath.c`:

   * Detect AVX2/FMA via `__builtin_cpu_supports("avx2")` on x86; fallback to NEON on aarch64.
   * Expose function pointers `cm_backend.vec_exp`, `cm_backend.vec_sin`, `cm_backend.vec_log`.
   * LOC: **\~80 LOC**

3. Add new API prototypes in `cmath.h`:

   ```c
   void cm_eval_vec(cm_expr *e, double *out, const double **vars, size_t n, cm_eval_mode_t mode);
   void cm_eval_vec_mt(...); // later
   ```

   * LOC: **\~20 LOC**

**Phase 0 total:** \~220 LOC

---

## Phase 1 — Vector kernels for hot functions (big wins)

**Goals:** Implement vectorized `exp`, `log`, `sin`, `cos`, `sqrt` kernels for AVX2 and NEON.

1. `vec_exp_avx2`, `vec_exp_neon` (skeleton above).

   * Implement bit-exponent trick for AVX2, lane `ldexp` for NEON initially.
   * Add high-quality polynomial successor later.
   * LOC: **\~250 LOC** (AVX2+NEON + helpers)

2. `vec_sin_avx2`, `vec_sin_neon` skeletons with quadrant logic placeholder.

   * Needs correct range reduction and quadrant masks later.
   * LOC: **\~300 LOC**

3. `vec_log` & `vec_pow` stubs (use standard `log` fallback initially), then implement faster approximations.

   * LOC: **\~150 LOC**

4. Test harness & microbenchmarks:

   * `bench/vec_exp_bench.c`, `bench/vec_sin_bench.c`
   * Unit accuracy checker vs `libm` over distributed input.
   * LOC: **\~250 LOC**

**Phase 1 total:** \~950 LOC

**Notes:** Expect major accuracy/edge-case work later (Payne–Hanek, special-case masking).

---

## Phase 2 — Bytecode & VM vector integration

**Goals:** Extend bytecode, compilation to emit vector ops, vector VM loop, minimize memory roundtrips.

1. Extend bytecode opcodes: add `OP_V_*` group.

   * Ensure `cm_compile()` can mark nodes vectorizable.
   * Add `cm_expr->bytecode_vector` and flags.
   * LOC: **\~200 LOC**

2. Implement vector VM:

   * Vector stack of `cm_vd` registers.
   * Ops implement register-to-register direct vector operations (no mem roundtrip).
   * Fallback scalar path for small `n`.
   * LOC: **\~400 LOC**

3. Implement compile-time pattern analysis to mark vectorizable subexpression nodes: `cm_pattern` extended.

   * Detect pure functions and vectorizable ops.
   * LOC: **\~150 LOC**

4. Tests: functional correctness (vector vs scalar results), performance microbench.

   * LOC: **\~200 LOC**

**Phase 2 total:** \~950 LOC

---

## Phase 3 — Numeric accuracy & special cases (quality)

**Goals:** Implement accurate range reduction, special-case mask handling, and accuracy modes.

1. Implement Payne–Hanek style range reduction for trig functions (hard part).

   * Use 128-bit or double-double arithmetic; library may implement small integer arithmetic for the constants.
   * LOC: **\~400 LOC**

2. Implement branchless special-case detection & masked blends (NaN, Inf, subnormal).

   * LOC: **\~150 LOC**

3. Add accuracy modes:

   * `CM_MODE_FAST`, `CM_MODE_BALANCED`, `CM_MODE_CORRECT`.
   * Switch between kernels / Newton refinement / double-double.
   * LOC: **\~120 LOC**

4. Add unit tests validating max error vs libm across ranges, and edge-case tests.

   * LOC: **\~300 LOC**

**Phase 3 total:** \~970 LOC

---

## Phase 4 — JIT & pattern-specialized codegen (optional, high payoff)

**Goals:** Emit native code for hot expressions (register-allocated, FMA usage, inlined constants).

1. Simple codegen backend (x86-64 + aarch64 optional): implement minimal assembler emitter for floating point registers + calling convention.

   * Use a tiny codegen layer; don't attempt full optimizer.
   * LOC: **\~1500 LOC**

2. Hook JIT into `cm_compile()` to JIT when expression evaluated many times.

   * Add eviction/caching policy inside `cm_expr_pool`.
   * LOC: **\~200 LOC**

3. Tests + safety (make sure `cm_free()` releases code, guard pages).

   * LOC: **\~200 LOC**

**Phase 4 total:** \~1900 LOC (big, optional)

---

## Phase 5 — Multithreaded batched evaluation & autotuning

**Goals:** Multithreading for very large batches, and simple runtime auto-tuner to choose kernel variants.

1. Implement `cm_eval_vec_mt()` with a small threadpool or `pthread` partitioning.

   * Use per-thread scratch buffers from `cm_expr_pool` to avoid locks.
   * LOC: **\~300 LOC**

2. Lightweight autotuner:

   * On first run run microbench of kernel variants (AVX2 vs scalar vs NEON) for the host CPU and record choices in `globalPool`.
   * LOC: **\~200 LOC**

3. Bench harness & results collection (CSV output).

   * LOC: **\~150 LOC**

**Phase 5 total:** \~650 LOC

---

## Phase 6 — Backend adapters & vendor integrations

**Goals:** Allow optional linking to SLEEF, MKL VML, AMD LibM, or OpenLibm.

1. Define `cm_backend_ops` structure with function pointers for `sin`, `exp`, `log`, `vectorized` variants.

   * Implement `cm_set_backend()` to register at runtime.
   * LOC: **\~120 LOC**

2. Implement adapters:

   * `backend_sleef.c` — thin wrapper calling SLEEF vector APIs.
   * `backend_mkl.c` — wrapper for MKL VML (if available).
   * Optional dynamic loading with `dlopen` to keep optional.
   * LOC each: **\~120 LOC** per backend + build scripts.

3. Tests to compare accuracy & perf between builtins and backends.

   * LOC: **\~200 LOC**

---

## QA, tests and total LOC summary (rough)

* Phase 0: 220
* Phase 1: 950
* Phase 2: 950
* Phase 3: 970
* Phase 4: 1900 (optional)
* Phase 5: 650
* Phase 6: 560 (approx)
* **Tests & Bench harness (overall):** 1200 (spread across phases)
* **TOTAL (without optional JIT):** \~5k LOC
* **TOTAL (with JIT):** \~7k LOC

> These LOC numbers are estimates to help your generator split tasks and judge complexity. They include comments and tests.

---

# Practical implementation guidance & checklists for your generative AI

Use this checklist when implementing each kernel or VM change. Each item is actionable and can be unit-tested.

1. **Implement the vector primitive layer** (`cm_vector.h`) — unit test `vec_set1/load/store/add/mul/fma` by round-tripping arrays.
2. **Implement `vec_poly_horner`** and stress-test it vs scalar Horner for random inputs.
3. **Implement `vec_exp_avx2`** and `vec_exp_neon` simple versions. Validate against `exp()`:

   * Test 100k random inputs across `[-700, 700]`.
   * Compute max absolute and relative error; plot error distribution.
4. **Add vector bytecode + compiler flags**:

   * Update `cm_compile()` so expression nodes that are pure & composed of supported ops are marked vectorizable.
   * Emit `OP_V_*` ops.
   * Add scalar fallback for tail elements and unsupported ops.
5. **Implement VM vector execution** avoiding memory roundtrips:

   * Provide `vec_op_*` functions that operate on `cm_vd` registers directly.
   * Keep interpreter loop with computed-goto (`goto *jump_table[op]`) for speed.
6. **Add accuracy modes**: `FAST` uses lower-degree polynomials, `BALANCED` uses degree tuned for 1 ulp < error < 1e-14 for common inputs, `CORRECT` uses double-double or Newton refinement for final rounding.
7. **Add autotuner** to pick best code path based on measured microbenchmarks on first run.
8. **Integrate a vendor backend** (SLEEF) as pluggable option: `cm_set_backend(&ops)` to use vendor vector functions when available.
9. **Add per-thread scratch buffers in pool** to remove locking in hot paths.
10. **Measure**: For each kernel produce microbench results (throughput, cycles per element, accuracy). Keep baseline `cmath` scalar timings to compare.

---

# Testing & benchmarking guidance

* **Unit tests**: small inputs, big inputs, edge-cases (NaN, Inf, subnormals), identity checks (e.g., `exp(log(x)) ~ x`).
* **Microbench**: throughput (evals/sec) for:

  * scalar `cm_eval()` baseline
  * `cm_eval_vec()` AVX2/NEON
  * `cm_eval_vec_mt()` for large `n`
* **Accuracy tests**: compare to `libm`, SLEEF, or MPFR for worst-case inputs.
* **Regression**: add CI to run a small accuracy & performance test on each PR.

---

# Final notes & suggested incremental plan for your generator

If you want a practical small-but-very-effective incremental work plan to hand to the generator (so it actually produces working code fast), tell it to implement these first:

1. **Add vector primitives and runtime dispatch** (Phase 0). — small, low risk, enables everything else.
2. **Implement AVX2 `vec_exp`** and test vs scalar `exp` for a practical speedup. Replace constants later. — big win.
3. **Wire vector bytecode + small vector VM** so `cm_eval_vec()` can evaluate simple expressions without roundtripping through memory. Use only `+ - * / exp` initially.
4. **Add `cm_eval_vec()` public API and microbench** to show speedups.
5. **Iterate: replace polynomial coefficients, add range reduction improvements, add branchless masks, add NEON parity.**

# Apple Silicon
Make sure **every part** of the roadmap and code skeletons explicitly includes **ARM64 NEON intrinsics** alongside **x86 AVX2/AVX-512**.

Here’s how we’ll do this:

---

## 🔹 Part 1 — Roadmap with Apple Silicon Integration

### Priority 0 — SIMD Vectorized Evaluation

* **Apple Silicon**:

  * Use **ARMv8.2-A NEON intrinsics** (`float64x2_t` for doubles, `float32x4_t` for floats).
  * Apple Silicon supports **full-width SIMD for doubles** (2 doubles per NEON register, 4 floats).
  * Use `vaddq_f64`, `vmulq_f64`, `vfmaq_f64` (FMA), `vrsqrteq_f64` + Newton-Raphson refine for reciprocal square root.
  * For trig/log/exp: start with SLEEF NEON kernels as reference, or implement lookup+poly approximations using NEON intrinsics.

* **x86 (Intel/AMD)**:

  * AVX2 (`__m256d` for 4 doubles) and AVX-512 (`__m512d` for 8 doubles).
  * Use `_mm256_fmadd_pd`, `_mm256_sqrt_pd`, `_mm256_set1_pd`, etc.
  * Dispatch at runtime with CPUID.

---

### Priority 1 — Polynomial Approximation + Estrin + Range Reduction

* **Both Apple Silicon & x86**:

  * Use **Estrin’s scheme** to evaluate polynomials with fewer dependent operations → better SIMD utilization.
  * Example: for `exp(x)`, range reduce with `x = k*ln2 + r` where `|r| < ln2/2`. Compute `exp(r)` with polynomial, then multiply by `2^k`.
  * Store coefficients in aligned arrays (64-byte aligned for cache and SIMD loads).

* **Apple Silicon**:

  * Use `vfmaq_f64` inside Estrin to fuse multiplications and adds.

---

### Priority 2 — CPU Dispatch

* On startup, detect **Apple Silicon NEON** vs **x86 AVX2/AVX-512**.
* Provide function pointers for kernels:

  ```c
  cm_backend_ops cm_ops;
  if (is_apple_silicon()) {
      cm_ops.sin = cm_sin_neon;
  } else if (cpu_has_avx2()) {
      cm_ops.sin = cm_sin_avx2;
  } else {
      cm_ops.sin = cm_sin_scalar;
  }
  ```
* This matches **MKL/Yeppp!/SLEEF’s runtime dispatch** model.

---

### Priority 3 — JIT Compilation

* For Apple Silicon: emit **ARM64 instructions** (`fmadd`, `fsub`, `ldr/str`, `bl` for function calls).
* For x86: emit **AVX2/AVX-512 machine code**.
* Cache JIT code in `cm_expr`.

---

## 🔹 Part 2 — C Code Skeletons with Apple Silicon + x86

## Example 1 — Vectorized Square Root (NEON vs AVX2)

```c
// Apple Silicon (ARM64 NEON)
#include <arm_neon.h>

static inline float64x2_t cm_sqrt_neon(float64x2_t x) {
    // Initial approximation
    float64x2_t approx = vrsqrteq_f64(x); // reciprocal sqrt estimate
    // Newton-Raphson refinement: y = y * (3 - x*y^2)/2
    approx = vmulq_f64(approx,
              vmulq_f64(vdupq_n_f64(0.5),
                vsubq_f64(vdupq_n_f64(3.0),
                          vmulq_f64(x, vmulq_f64(approx, approx)))));
    return vmulq_f64(x, approx); // sqrt(x) = x * rsqrt(x)
}
```

```c
// x86 AVX2
#include <immintrin.h>

static inline __m256d cm_sqrt_avx2(__m256d x) {
    return _mm256_sqrt_pd(x); // direct AVX2 sqrt
}
```

---

## Example 2 — Vectorized Exponential via Range Reduction (Estrin)

```c
// Coefficients for exp(r) polynomial on [-ln2/2, ln2/2]
static const double C[] __attribute__((aligned(64))) = {
    1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0
};

// Apple Silicon NEON
static inline float64x2_t cm_exp_neon(float64x2_t x) {
    // Range reduction: x = k*ln2 + r
    float64x2_t inv_ln2 = vdupq_n_f64(1.4426950408889634); // 1/ln2
    float64x2_t k_real = vfmaq_f64(vdupq_n_f64(0.0), x, inv_ln2);
    int64x2_t k = vcvtq_s64_f64(k_real); // integer k
    float64x2_t r = vsubq_f64(x, vmulq_f64(vcvtq_f64_s64(k), vdupq_n_f64(0.6931471805599453)));

    // Estrin polynomial for exp(r)
    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t p = vfmaq_f64(vdupq_n_f64(C[0]), r, vdupq_n_f64(C[1]));
    p = vfmaq_f64(p, r2, vdupq_n_f64(C[2]));
    p = vfmaq_f64(p, vmulq_f64(r, r2), vdupq_n_f64(C[3]));
    p = vfmaq_f64(p, vmulq_f64(r2, r2), vdupq_n_f64(C[4]));

    // Scale by 2^k
    int64x2_t two_k = vshlq_s64(vdupq_n_s64(1), k);
    float64x2_t scale = vreinterpretq_f64_s64(two_k);
    return vmulq_f64(p, scale);
}
```

```c
// x86 AVX2
static inline __m256d cm_exp_avx2(__m256d x) {
    // Similar approach: range reduction + Estrin + AVX2 intrinsics
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
    const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634);

    __m256d k_real = _mm256_mul_pd(x, inv_ln2);
    __m256d k_round = _mm256_round_pd(k_real, _MM_FROUND_TO_NEAREST_INT);
    __m256d r = _mm256_fnmadd_pd(k_round, ln2, x);

    // Polynomial
    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d p = _mm256_add_pd(_mm256_set1_pd(1.0),
               _mm256_fmadd_pd(r, _mm256_set1_pd(1.0),
               _mm256_fmadd_pd(r2, _mm256_set1_pd(0.5),
               _mm256_fmadd_pd(_mm256_mul_pd(r, r2), _mm256_set1_pd(1.0/6.0),
               _mm256_fmadd_pd(_mm256_mul_pd(r2, r2), _mm256_set1_pd(1.0/24.0),
                               _mm256_mul_pd(r2, _mm256_mul_pd(r2, r)) // ~1/120
                               ))))));
    // Scale by 2^k (convert k to int)
    __m256i ki = _mm256_cvtpd_epi32(k_round);
    __m256d scale = _mm256_castsi256_pd(_mm256_slli_epi64(ki, 52)); // 2^k
    return _mm256_mul_pd(p, scale);
}
```

---

## Example 3 — Vectorized Evaluation API (Apple Silicon + x86)

```c
// Batch API: Evaluate expression for N inputs in parallel
void cm_eval_vec(cm_expr *e, double *out, const double **vars, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    // Apple Silicon NEON (2 doubles at a time)
    for (; i + 1 < n; i += 2) {
        float64x2_t x = vld1q_f64(&vars[0][i]);
        float64x2_t y = vld1q_f64(&vars[1][i]);
        float64x2_t res = cm_sqrt_neon(vmlaq_f64(vmulq_f64(x,x), y,y));
        vst1q_f64(&out[i], res);
    }
#elif defined(__AVX2__)
    // x86 AVX2 (4 doubles at a time)
    for (; i + 3 < n; i += 4) {
        __m256d x = _mm256_loadu_pd(&vars[0][i]);
        __m256d y = _mm256_loadu_pd(&vars[1][i]);
        __m256d res = _mm256_sqrt_pd(_mm256_fmadd_pd(y, y, _mm256_mul_pd(x, x)));
        _mm256_storeu_pd(&out[i], res);
    }
#endif
    // Fallback scalar path
    for (; i < n; i++) {
        double xv = vars[0][i];
        double yv = vars[1][i];
        out[i] = sqrt(xv*xv + yv*yv);
    }
}
```

---

## 🔹 Part 3 — Next Steps for CMath on Apple Silicon

1. **Start with NEON intrinsics for exp, log, sin, cos, sqrt** using range reduction + polynomial approximation.
2. Add **runtime detection** so you can build **fat binaries**:

   * Apple: default to NEON (always available).
   * x86: dispatch AVX-512 → AVX2 → SSE2 → scalar.
3. Expand your **bytecode VM** with **vector opcodes** (operate on SIMD registers instead of scalars).
4. Add **accuracy/performance modes** (`CM_FAST`, `CM_BALANCED`, `CM_CORRECT`) that select different approximation polynomials.
5. Later: **JIT compile ARM64 NEON code** into `cm_expr` for hot expressions.


# Coding instructions
Do not add unnecessary comments. The comments should not have obvious comments which would seem redundant to a skilled programmer. Clear out any obvious comments. Every comment should have lower case first letter.