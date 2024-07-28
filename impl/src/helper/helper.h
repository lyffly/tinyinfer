#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include "cublasLt.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}