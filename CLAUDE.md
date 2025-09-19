# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CMath is a lightweight recursive descent parser and mathematical evaluation engine written in ANSI C. It provides runtime evaluation of mathematical expressions with support for variables, standard mathematical functions, and thread-safe operation.

## Core Architecture

### Single File Library Structure
- **cmath.h**: Main header defining the API and data structures
- **cmath.c**: Complete implementation of the parser and evaluator
- **example.c**: Simple usage demonstration
- **benchmark.c**: Performance comparison tool

### Key Data Structures
- `cm_expr`: Expression tree node representing parsed mathematical expressions
- `cm_variable`: Variable binding structure for runtime variable resolution
- `cm_expr_pool`: Thread-safe memory pool for expression node allocation with mutex protection

### Threading Model
The library implements thread-safety through:
- Global expression pool with pthread mutex protection (`globalPool`)
- Memory compaction and reuse strategy to avoid fragmentation
- Pool-based allocation system in `cmath.c:30-37`

## Build System

### CMake Build
```bash
mkdir -p build
cd build
cmake ..
make
```

**Note**: The current build has compilation errors that need resolution before the library can be built successfully. The main issues are:
- Function signature mismatches between header and implementation
- Missing function declarations
- Incompatible pointer type assignments

### Direct Compilation
For simple compilation without CMake:
```bash
gcc -o example example.c cmath.c -lm -lpthread
gcc -o benchmark benchmark.c cmath.c -lm -lpthread
```

## API Usage Patterns

### Primary Functions
1. `cm_interp()`: One-shot expression evaluation
2. `cm_compile()`: Parse expression with variable bindings  
3. `cm_eval()`: Evaluate pre-compiled expression
4. `cm_free()`: Release expression memory

### Common Usage Pattern
```c
// Compile once, evaluate many times
double x, y;
cm_variable vars[] = {{"x", &x}, {"y", &y}};
cm_expr *expr = cm_compile("sqrt(x^2+y^2)", vars, 2, &err);

// Efficient repeated evaluation
x = 3; y = 4;
double result1 = cm_eval(expr);
x = 5; y = 12; 
double result2 = cm_eval(expr);

cm_free(expr);
```

## Parser Implementation

### Grammar Structure
- Recursive descent parser implementing standard operator precedence
- Left-to-right exponentiation evaluation (differs from mathematical convention)
- Support for standard C math functions (sin, cos, sqrt, log, etc.)
- Constants: pi, e

### Memory Management
- Pool-based allocation with compaction strategy
- Thread-safe node allocation and deallocation
- Automatic memory cleanup through `cm_free()`

## Development Notes

### Code Issues to Address
The codebase currently has compilation errors that prevent building:
- Header/implementation signature mismatches in `cm_eval()`
- Missing `new_expr()` function declarations
- Pointer type compatibility issues in pool management

### Performance Characteristics
- Designed for high-performance repeated evaluation
- Benchmark shows significant speed improvements over native C in some cases
- Memory-efficient through expression tree reuse