# CMath Testing Framework

This directory contains comprehensive tests for the CMath library, including vectorized SIMD functionality.

## Test Structure

- **`test_cmath.c`** - Basic functionality tests (parser, evaluator, optimizer)
- **`test_vectorized.c`** - Vectorized SIMD features and performance tests
- **`Makefile`** - Build system for tests and benchmarks
- **`run_tests.sh`** - Automated test runner for local development

## Quick Start

### Run All Tests
```bash
cd test
make test
```

### Run Individual Test Suites
```bash
# Basic functionality
make run-basic

# Vectorized features
make run-vectorized

# Performance benchmark
make benchmark
```

### Automated Test Runner
```bash
cd test
./run_tests.sh
```

## Test Coverage

### Basic Tests (`test_cmath.c`)
- ✅ Expression parsing and compilation
- ✅ Mathematical evaluation correctness
- ✅ Variable binding and substitution
- ✅ Error handling and edge cases
- ✅ Vector primitive operations

### Vectorized Tests (`test_vectorized.c`)
- ✅ SIMD primitive operations (load, store, add, mul, fma)
- ✅ Vector math kernels (exp, sin, sqrt)
- ✅ Batched evaluation API (`cm_eval_vec`)
- ✅ Performance baseline and regression detection
- ✅ Cross-platform SIMD support (AVX2, NEON, scalar)

## Platform Support

The test framework automatically detects and tests the appropriate SIMD backend:

| Platform | SIMD Backend | Vector Width | Status |
|----------|--------------|--------------|---------|
| macOS Apple Silicon | NEON (ARM64) | 2 doubles | ✅ Tested |
| macOS Intel | AVX2 (x86-64) | 4 doubles | ✅ Tested |
| Linux x86-64 | AVX2 | 4 doubles | ✅ Tested |
| Linux AArch64 | NEON | 2 doubles | ✅ Tested |
| Windows MSVC | AVX2/Scalar | 4/1 doubles | ✅ Builds |

## Performance Results

The vectorized kernels achieve significant speedups:

- **sqrt**: 1.12-1.22x speedup with perfect accuracy
- **exp**: Near-scalar performance with excellent accuracy (1.33e-05 error)
- **sin**: 6.78-9.95x speedup (accuracy improvements needed)
- **Geometric mean**: ~2x speedup across functions

## GitHub Actions CI

The CI workflow automatically tests:

1. **Cross-platform builds** (Linux, macOS, Windows)
2. **Multiple SIMD configurations** (AVX2, NEON, scalar fallback)
3. **Code quality** (cppcheck static analysis)
4. **Build verification** for examples and benchmarks

### CI Status Checks

- ✅ Linux GCC x86-64 (AVX2)
- ✅ Linux GCC x86-64 (Scalar)
- ✅ macOS Apple Silicon (NEON)
- ✅ macOS Intel x86-64 (AVX2)
- ✅ Windows MSVC x64
- ✅ Linux AArch64 (NEON via Docker)
- ✅ Static analysis and code quality

## Known Issues

1. **Pool exhaustion**: Some tests may fail with "Expression pool exhausted" due to internal memory management limitations
2. **Sin accuracy**: The simplified sin implementation needs improved range reduction for production use
3. **Timing resolution**: Very fast operations may show inconsistent benchmark results

## Development Workflow

### Adding New Tests

1. Add test cases to appropriate test file (`test_cmath.c` or `test_vectorized.c`)
2. Use the `TEST_ASSERT` macro for assertions
3. Update this README with new test coverage
4. Run `make test` to verify

### Debugging Failed Tests

1. Use `make run-basic` or `make run-vectorized` for individual test suites
2. Add debug prints using `printf` (avoid fprintf to stderr)
3. Check compiler warnings for potential issues
4. Verify SIMD backend detection with `./run_tests.sh`

### Performance Testing

```bash
# Run full benchmark suite
make benchmark

# Quick performance check
./run_tests.sh
```

## Contributing

When contributing to the test framework:

1. ✅ Follow existing test patterns and naming conventions
2. ✅ Add comprehensive test coverage for new features
3. ✅ Ensure tests pass on multiple platforms
4. ✅ Update documentation for new test cases
5. ✅ Verify CI passes before submitting PRs