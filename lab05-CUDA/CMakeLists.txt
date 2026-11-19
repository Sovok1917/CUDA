cmake_minimum_required(VERSION 3.18)
project(lab05_cuda LANGUAGES CXX CUDA)

add_executable(lab05_cuda main.cu)

set_target_properties(lab05_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(lab05_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --fmad=false>
)
