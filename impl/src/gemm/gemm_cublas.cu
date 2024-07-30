#include <iostream>
#include "../../include/kernels.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../helper/helper.h"

// m*n + m*n = m*n
// or m*n + 1*n = m*n
template <typename T>
__global__ void add_bias_fp(T* inout, T* bias, int m, int n, bool is_boardcast, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto inout_ptr = reinterpret_cast<T*>(inout);
    auto bias_ptr = reinterpret_cast<T*>(bias);
    if ((index < length) and is_boardcast) {
        int index_bias = index % n;
        inout_ptr[index] = inout_ptr[index] + bias_ptr[index_bias];
    } else if ((index < length) and (!is_boardcast)) {
        inout_ptr[index] = inout_ptr[index] + bias_ptr[index];
    }
}

static cublasHandle_t cublas_handle;

void cublas_gemm_fp16_v1(cublasHandle_t handle, bool trsA, bool trsB, int m, int n, int k,
                         const float* alpha, void* A, void* B, const float* beta, void* C) {
    // C = A * B    m*k k*n = m*n  1*512 512*1000 = 1*1000
    // CT = BT *AT  n*k k*m = n*m  1000*512 512*1 = 1000*1
    // CT = B * AT  k*n k*m = n*m  512*1000 512*1 = 1000*1
    half h_alpha = __float2half(*alpha);
    // half h_beta = __float2half(*beta);
    half h_beta = __float2half(0.0f);
    cublasOperation_t transa_ = CUBLAS_OP_N;
    cublasOperation_t transb_ = CUBLAS_OP_N;
    int m_ = n;      // 1000
    int n_ = m;      // 1
    int k_ = k;      // 512
    int lda_ = n;    // 1000 col major
    int ldb_ = k;    // 512
    int ldc_ = n;    // 1000
    if (trsB) {
        transa_ = CUBLAS_OP_T;
        lda_ = k;
    }
    if (trsA) {
        transb_ = CUBLAS_OP_T;
        ldb_ = m;
    }
    // printf("m %d k %d n  %d \n",m_,k_,n_);
    // printf("lda= %d ldb= %d \n",lda_, ldb_);
    // cudaMemset((void**)&C, 0,  m*n*sizeof(half));
    checkCublasStatus(cublasHgemm(handle, transa_, transb_, m_, n_, k_, &h_alpha, (__half*)B, lda_,
                                  (__half*)A, ldb_, &h_beta, (__half*)C, ldc_));
    // add bias
}

void cublas_gemm_fp32_v1(cublasHandle_t handle, bool trsA, bool trsB, int m, int n, int k,
                         const float* alpha, void* A, void* B, const float* beta, void* C) {
    cublasOperation_t transa_ = CUBLAS_OP_N;
    cublasOperation_t transb_ = CUBLAS_OP_N;
    int m_ = n;      // 1000
    int n_ = m;      // 1
    int k_ = k;      // 512
    int lda_ = n;    // 1000 col major
    int ldb_ = k;    // 512
    int ldc_ = n;    // 1000
    float beta0 = 0.0f;
    if (trsB) {
        transa_ = CUBLAS_OP_T;
        lda_ = k;
    }
    if (trsA) {
        transb_ = CUBLAS_OP_T;
        ldb_ = m;
    }
    // printf("m %d k %d n  %d \n",m_,k_,n_);
    // printf("lda= %d ldb= %d \n",lda_, ldb_);
    // cudaMemset((void**)&C, 0,  m*n*sizeof(half));
    checkCublasStatus(cublasSgemm(handle, transa_, transb_, m_, n_, k_, alpha, (float*)B, lda_,
                                  (float*)A, ldb_, &beta0, (float*)C, ldc_));
    // add bias
}

bool gemm_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                  int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                  bool transA, bool transB, std::vector<int> in_shape,
                  std::vector<int> weight_shape, std::vector<int> bias_shape,
                  std::vector<int> out_shape, std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    // 2d tensor only, batch gemm todo
    // A=N ; B=N or T
    int m = in_shape.at(0);
    int k = in_shape.at(1);
    int n = 0;

    if (transB) {    // n*k
        n = weight_shape.at(0);
    } else {    // k*n
        n = weight_shape.at(1);
    }
    // printf("alpha %f beta %f \n",alpha, beta);
    // printf("m %d k %d n  %d \n",m,k,n);
    if (!cublas_handle) {
        cublasCreate_v2(&cublas_handle);
    }
    cublasSetStream(cublas_handle, stream);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    if (dtype == "float32") {
        cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        cublas_gemm_fp32_v1(cublas_handle, transA, transB, m, n, k, &alpha, (void*)in_ptr,
                            (void*)weight_ptr, &beta, (void*)out_ptr);
        bool is_boardcast = bias_shape.size() == 1 ? true : false;
        int block_size = 512;
        int length = 1;
        for (auto& shape : out_shape) {
            length *= shape;
        }
        int grid_size = (length + block_size - 1) / block_size;
        add_bias_fp<float><<<grid_size, block_size, 0, stream>>>((float*)out_ptr, (float*)bias_ptr,
                                                                 m, n, is_boardcast, length);
    } else if (dtype == "float16") {
        cublas_gemm_fp16_v1(cublas_handle, transA, transB, m, n, k, &alpha, (void*)in_ptr,
                            (void*)weight_ptr, &beta, (void*)out_ptr);
        bool is_boardcast = bias_shape.size() == 1 ? true : false;
        int block_size = 512;
        int length = 1;
        for (auto& shape : out_shape) {
            length *= shape;
        }
        int grid_size = (length + block_size - 1) / block_size;
        add_bias_fp<half><<<grid_size, block_size, 0, stream>>>((half*)out_ptr, (half*)bias_ptr, m,
                                                                n, is_boardcast, length);
    } else {
        printf("Gemm not support data type %s !!! \n", dtype.c_str());
    }

    return true;
}
