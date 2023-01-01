cmake_minimum_required(VERSION 3.16)

add_subdirectory(3rdparty/cnpy)

project(dgsparse C CXX CUDA)
find_package( CUDAToolkit )
set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++14")

file(GLOB DGSPARSE_SRC
	src/ge-spmm/*.cu
  src/ge-spmm/*.cc
  src/sddmm/*.cu
)

add_library(dgsparse SHARED ${DGSPARSE_SRC})
target_include_directories(dgsparse PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dgsparse PRIVATE CUDA::cusparse CUDA::cublas CUDA::curand CUDA::cudart cnpy)
install(TARGETS dgsparse)

add_subdirectory(example/ge-spmm)
add_subdirectory(example/sddmm)
