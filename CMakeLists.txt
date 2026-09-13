cmake_minimum_required(VERSION 3.18)
project(HighPerformanceComputing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce -O2 optimization and architectural baseline
add_compile_options(-O2 -msse2)

add_subdirectory(common)
add_subdirectory(lab01-CUDA)
