include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories("/usr/include")
include_directories("/usr/local/cuda/include")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../3rd/cutlass/include)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

# cuda lib build
file(GLOB SRCS_IMPL ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cu 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp)
message(${SRCS_IMPL})
add_library(cuda_kernels STATIC ${SRCS_IMPL})
target_link_libraries(cuda_kernels PRIVATE cuda cudart cublasLt cublas cudnn)
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++17)
set(__CUTLASS_CUDA_NVCC_FLAGS ${CUTLASS_CUDA_NVCC_FLAGS} CACHE INTERNAL "")
set(CUDA_COMPILE_LANGUAGE CUDA)
set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_NVCC_FLAGS})
target_compile_options(
    cuda_kernels
    PRIVATE
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
    )

# c++ pybind11 lib build
file(GLOB SRCS_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(kernels  ${SRCS_INTERFACE})

# link
link_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(kernels PRIVATE pybind11::module pybind11::embed cuda_kernels)
set_target_properties(kernels PROPERTIES    PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")


install(TARGETS kernels LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
