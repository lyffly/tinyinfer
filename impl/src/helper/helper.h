#pragma once

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "cudnn.h"


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

struct ConvDesc {
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnActivationDescriptor_t activation_desc;
};

struct PoolDesc {
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnPoolingDescriptor_t pooling_desc;
};

struct Handles {
    cudnnHandle_t cudnn_handle;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
};
