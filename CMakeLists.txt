cmake_minimum_required(VERSION 3.18)
project(HighPerformanceComputing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_compile_options(-O2 -msse2 -g -mfma)

add_subdirectory(lab01-CUDA)
