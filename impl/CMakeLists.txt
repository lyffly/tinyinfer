include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})

# cuda lib build
add_library(cuda_kernels STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/elementwise/elementwise.cu)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# c++ pybind11 lib build
pybind11_add_module(kernels ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp)

# link
link_directories(${CMAKE_CURRENT_BINARY_DIR})   # 暴露库路径
target_link_libraries(kernels PRIVATE pybind11::module pybind11::embed cuda_kernels)
set_target_properties(kernels PROPERTIES    PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")


# install(TARGETS cuda_kernels LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS kernels LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

