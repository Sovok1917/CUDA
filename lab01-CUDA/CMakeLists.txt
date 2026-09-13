add_executable(lab01-CUDA
    main.cpp
    matmul.cpp
)

target_link_libraries(lab01-CUDA PRIVATE common)
