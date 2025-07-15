cmake_minimum_required(VERSION 3.20)
project(coo_merge_project LANGUAGES C CXX CUDA)

enable_language(CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target H200 GPU (Hopper architecture)
set(CMAKE_CUDA_ARCHITECTURES 90)

# Separate compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# NVCOMP paths
set(NVCOMP_INCLUDE "$ENV{BASE_DIR}/install/nvcomp-install/usr/include/nvcomp_12")
set(NVCOMP_LIB "$ENV{BASE_DIR}/install/nvcomp-install/usr/lib/x86_64-linux-gnu/nvcomp/12")

add_executable(run
    main.cu
    processing.cu 
    ip_anonymize.cu
    cryptopan.cu
    matrix.cu 
    compress.cu
    write.c 
    config.h
)

# Include + Link NVCOMP
target_include_directories(run PRIVATE ${NVCOMP_INCLUDE})
link_directories(${NVCOMP_LIB})
target_link_libraries(run PRIVATE nvcomp)

# 编译属性
set_target_properties(run PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# 编译优化选项
target_compile_options(run PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas=-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:C>:-O3>
)

# 可选：开启调试宏
target_compile_definitions(run PRIVATE -DCUDA_VERBOSE=1)
