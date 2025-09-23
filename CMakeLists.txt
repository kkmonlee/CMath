cmake_minimum_required(VERSION 3.15)
project(CMath LANGUAGES C CXX)

option(CM_FAST_MATH  "Enable fast-math (may change FP semantics)" ON)
option(CM_ARCH_NATIVE "Build with -march=native" ON)
option(CM_ENABLE_LTO "Enable LTO/IPO" ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# LLVM (ORC JIT)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Sources
add_library(cmath
        cmath.c
        cmath_jit_llvm.cpp
        cmath_jit_llvm.cpp
        cmath_jit_llvm.h    # C++ ORC-LLJIT bridge
)
set_source_files_properties(cmath_jit_llvm.cpp PROPERTIES LANGUAGE CXX)

if(TARGET LLVM::LLVM)
    target_link_libraries(cmath PRIVATE LLVM::LLVM)
else()
    # Fallback
    target_link_libraries(cmath PRIVATE LLVM)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(cmath PRIVATE m)
endif()

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark PRIVATE cmath)

# Optimization flags
foreach(tgt IN ITEMS cmath benchmark)
    target_compile_options(benchmark PRIVATE -O3 -ffast-math -fno-math-errno -ffp-contract=fast
            -Rpass=loop-vectorize -Rpass-missed=loop-vectorize)

    if(MSVC)
        target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:/O2 /fp:fast /DNDEBUG>)
    else()
        target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-math-errno -fno-trapping-math -ffp-contract=fast>)
        if(CM_FAST_MATH)
            target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-ffast-math>)
        endif()
        if(CM_ARCH_NATIVE)
            target_compile_options(${tgt} PRIVATE $<$<CONFIG:Release>:-march=native>)
        endif()
    endif()
endforeach()

if(APPLE)
    execute_process(COMMAND brew --prefix llvm OUTPUT_VARIABLE BREW_LLVM_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(BREW_LLVM_PREFIX)
        set(CMAKE_BUILD_RPATH "${BREW_LLVM_PREFIX}/lib")
    endif()
endif()

# IPO/LTO
if(CM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
    if(ipo_ok)
        set_property(TARGET cmath benchmark PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO not enabled: ${ipo_msg}")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()