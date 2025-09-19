cmake_minimum_required(VERSION 3.10)
project(CMath)

set(CMAKE_C_STANDARD 99)

# Create library from cmath.c
add_library(cmath cmath.c)
target_link_libraries(cmath m pthread)

# Create example executable
add_executable(example example.c)
target_link_libraries(example cmath)

# Create benchmark executable
add_executable(benchmark benchmark.c)
target_link_libraries(benchmark cmath)