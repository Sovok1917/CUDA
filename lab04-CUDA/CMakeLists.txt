cmake_minimum_required(VERSION 3.18)
project(lab04_cuda LANGUAGES CXX CUDA)

add_executable(lab04_cuda main.cu)

set_target_properties(lab04_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(lab04_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --fmad=false>
)
