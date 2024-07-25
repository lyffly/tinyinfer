cmake_minimum_required(VERSION 3.20)

project(kernels LANGUAGES CXX CUDA)

# c++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g ")

# cuda flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -g")

# 寻找CUDA工具包
find_package(CUDA REQUIRED)

add_subdirectory(3rd/pybind11)

# 支持cuda架构sm_75;sm_80
set(CMAKE_CUDA_ARCHITECTURES 75;80;86)

add_subdirectory(impl)

# 指定安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/tinyinfer CACHE INTERNAL "install directories")

