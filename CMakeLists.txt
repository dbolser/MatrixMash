cmake_minimum_required(VERSION 3.18)
project(MatrixMash CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit (provides CUDA::cudart target)
find_package(CUDAToolkit REQUIRED)

# CUDA architecture detection
# CUDA_ARCH should be just the number (e.g., 86 for sm_86)
if(NOT DEFINED CUDA_ARCH)
    # Default to 86 (RTX 4060 Laptop, sm_86)
    set(CUDA_ARCH "86" CACHE STRING "CUDA architecture (e.g., 86 for sm_86)")
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# Source files
set(CUDA_SOURCES distance_cuda.cu)

# Build shared library (for Python wrapper)
add_library(distance_cuda_shared SHARED ${CUDA_SOURCES})
set_target_properties(distance_cuda_shared PROPERTIES
    OUTPUT_NAME "distance_cuda"
    PREFIX ""
    SUFFIX ".so"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCH}
)
target_link_libraries(distance_cuda_shared PRIVATE CUDA::cudart)

# Build static library (for benchmarks)
add_library(distance_cuda_static STATIC ${CUDA_SOURCES})
set_target_properties(distance_cuda_static PROPERTIES
    OUTPUT_NAME "distance_cuda_static"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCH}
)
target_link_libraries(distance_cuda_static PRIVATE CUDA::cudart)

# Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Benchmark executable
add_executable(distance_cuda_benchmark distance_cuda_benchmark.cpp)
target_link_libraries(distance_cuda_benchmark 
    PRIVATE 
    distance_cuda_static
    benchmark::benchmark
    benchmark::benchmark_main
)

# Install rules (optional)
install(TARGETS distance_cuda_shared
    LIBRARY DESTINATION lib
)

