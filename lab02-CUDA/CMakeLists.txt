add_executable(lab02-CUDA main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(lab02-CUDA PRIVATE OpenMP::OpenMP_CXX)
