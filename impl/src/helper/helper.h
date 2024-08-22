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


/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                           \
    {                                                                                   \
        cutlass::Status error = status;                                                 \
        if (error != cutlass::Status::kSuccess) {                                       \
            std::cerr << "[Error] Got cutlass error: " << cutlassGetStatusString(error) \
                      << " at: " << __LINE__ << std::endl;                              \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                            \
    {                                                                                 \
        cudaError_t error = status;                                                   \
        if (error != cudaSuccess) {                                                   \
            std::cerr << "[Error] Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;                       \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    }

struct GpuTimer;

struct ConvDesc {
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnActivationDescriptor_t activation_desc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;
    void* workspace_ptr;
    cudnnHandle_t cudnn_handle;
    cudaStream_t cuda_stream;
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

#define WARP_SIZE 32


__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}