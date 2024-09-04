cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

project(flashatt-libs LANGUAGES CXX CUDA)

set(LIB_ROOT ${PROJECT_SOURCE_DIR})
message("LIB_ROOT = ${LIB_ROOT}")
set(CUTLASS_ROOT ${LIB_ROOT}/../3rd/cutlass)
message("CUTLASS_ROOT = ${CUTLASS_ROOT}")

add_subdirectory(flash_attn)
