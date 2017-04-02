cmake_minimum_required(VERSION 3.6)
project(CMath)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES cmath.h cmath.c benchmark.c example.c)
add_executable(CMath ${SOURCE_FILES})