include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})

# cuda lib build
file(GLOB SRCS_IMPL ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp)
message(${SRCS_IMPL})
add_library(cuda_kernels STATIC ${SRCS_IMPL})

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# c++ pybind11 lib build
file(GLOB SRCS_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(kernels  ${SRCS_INTERFACE})

# link
link_directories(${CMAKE_CURRENT_BINARY_DIR})   # 暴露库路径
target_link_libraries(kernels PRIVATE pybind11::module pybind11::embed cuda_kernels)
set_target_properties(kernels PROPERTIES    PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")


install(TARGETS kernels LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

