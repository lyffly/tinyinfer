cmake_minimum_required(VERSION 3.10)

project(kernels LANGUAGES CXX)
add_subdirectory(3rd/pybind11)

pybind11_add_module(kernels impl/interface.cpp)


