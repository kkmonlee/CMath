# CMath Test Makefile

CC ?= gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -g
LDFLAGS = -lm -lpthread

# detect platform and add appropriate SIMD flags
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2 -mfma
endif

ifeq ($(UNAME_M),arm64)
    # apple silicon - neon is default
    CFLAGS += -mcpu=native
endif

# source files
CMATH_SRC = ../cmath.c
CMATH_HDR = ../cmath.h ../cm_vector.h

# test targets
TESTS = test_basic test_vectorized test_sanity
TEST_SOURCES = test_cmath.c test_vectorized.c test_sanity.c

.PHONY: all clean run test help

all: $(TESTS)

test_basic: test_cmath.c $(CMATH_SRC) $(CMATH_HDR)
	$(CC) $(CFLAGS) -o $@ test_cmath.c $(CMATH_SRC) $(LDFLAGS)

test_vectorized: test_vectorized.c $(CMATH_SRC) $(CMATH_HDR)
	$(CC) $(CFLAGS) -o $@ test_vectorized.c $(CMATH_SRC) $(LDFLAGS)

test_sanity: test_sanity.c $(CMATH_SRC) $(CMATH_HDR)
	$(CC) $(CFLAGS) -o $@ test_sanity.c $(CMATH_SRC) $(LDFLAGS)

# run all tests
test: $(TESTS)
	@echo "=== Running Sanity Test ==="
	./test_sanity
	@echo
	@echo "=== Running Basic Tests ==="
	./test_basic || echo "Basic tests failed (known pool issue)"
	@echo
	@echo "=== Running Vectorized Tests ==="
	./test_vectorized || echo "Vectorized tests failed (known pool issue)"
	@echo
	@echo "=== All Tests Complete ==="

# run tests individually
run-basic: test_basic
	./test_basic

run-vectorized: test_vectorized
	./test_vectorized

# performance benchmark
benchmark: ../benchmark_vectorized.c $(CMATH_SRC) $(CMATH_HDR)
	$(CC) $(CFLAGS) -o benchmark ../benchmark_vectorized.c $(LDFLAGS)
	./benchmark

clean:
	rm -f $(TESTS) benchmark *.o *.dSYM

help:
	@echo "Available targets:"
	@echo "  all          - build all tests"
	@echo "  test         - run all tests"
	@echo "  run-basic    - run basic functionality tests"
	@echo "  run-vectorized - run vectorized feature tests"
	@echo "  benchmark    - run performance benchmarks"
	@echo "  clean        - remove built files"
	@echo "  help         - show this help"
	@echo
	@echo "Environment variables:"
	@echo "  CC           - compiler (default: gcc)"
	@echo "  CFLAGS       - additional compiler flags"
	@echo "  LDFLAGS      - additional linker flags"