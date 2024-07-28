#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cublasLt.h"
#include "cublas_v2.h"
#include "stdio.h"
#include "math.h"
#include <iostream>

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include "../helper/helper.h"


void cublas_gemm_fp16_v1(cublasHandle_t handle, bool trsA, bool trsB,
                        int m, int n, int k,
                        const float *alpha,
                        void* A,
                        void* B,
                        const float *beta,
                        void* C) {
    // C = A * B    m*k k*n = m*n  1*512 512*1000 = 1*1000
    // CT = BT *AT  n*k k*m = n*m  1000*512 512*1 = 1000*1
    // CT = B * AT  k*n k*m = n*m  512*1000 512*1 = 1000*1
    half h_alpha = __float2half(*alpha);
    // half h_beta = __float2half(*beta);
    half h_beta = __float2half(0.0f);
    cublasOperation_t transa_ = CUBLAS_OP_N;
    cublasOperation_t transb_ = CUBLAS_OP_N;
    int m_ = n; //1000
    int n_ = m; //1
    int k_ = k; //512
    int lda_ = n; // 1000 col major
    int ldb_ = k; // 512 
    int ldc_ = n; // 1000 
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
    //cudaMemset((void**)&C, 0,  m*n*sizeof(half));
    checkCublasStatus(cublasHgemm(handle, transa_, transb_,
                        m_, n_, k_,
                        &h_alpha,
                        (__half*)B, lda_,
                        (__half*)A, ldb_,
                        &h_beta,
                        (__half*)C, ldc_));
    // add bias

}

void cublas_gemm_fp32_v1(cublasHandle_t handle, bool trsA, bool trsB,
                        int m, int n, int k,
                        const float *alpha,
                        void* A,
                        void* B,
                        const float *beta,
                        void* C) {
    // half h_alpha = __float2half(*alpha);
    // half h_beta = __float2half(*beta);
    // half h_beta = __float2half(0.0f);
    cublasOperation_t transa_ = CUBLAS_OP_N;
    cublasOperation_t transb_ = CUBLAS_OP_N;
    int m_ = n; //1000
    int n_ = m; //1
    int k_ = k; //512
    int lda_ = n; // 1000 col major
    int ldb_ = k; // 512 
    int ldc_ = n; // 1000 
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
    //cudaMemset((void**)&C, 0,  m*n*sizeof(half));
    checkCublasStatus(cublasSgemm(handle, transa_, transb_,
                        m_, n_, k_,
                        alpha,
                        (float*)B, lda_,
                        (float*)A, ldb_,
                        beta,
                        (float*)C, ldc_));
    // add bias
}


bool gemm_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr, int64_t workspace_size, 
                int64_t workspace_ptr, float alpha, float beta, bool transA, bool transB, 
                std::vector<int> in_shape, std::vector<int> weight_shape, std::vector<int> bias_shape,
                std::vector<int> out_shape, std::string dtype) {
    
    // A=N ; B=N or T
    int m = in_shape.at(0);
    int k = in_shape.at(1);
    int n = 0;
    
    if (transB) { // n*k
        n = weight_shape.at(0);
    }else{ // k*n
        n = weight_shape.at(1);
    }
    // printf("alpha %f beta %f \n",alpha, beta);
    // printf("m %d k %d n  %d \n",m,k,n);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (dtype == "float32") {
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        cublas_gemm_fp32_v1(handle, transA, transB,
                        m, n, k,
                        &alpha,
                        (void*)in_ptr, 
                        (void*)weight_ptr, 
                        &beta,
                        (void*)out_ptr);
    } else if (dtype == "float16") {
        cublas_gemm_fp16_v1(handle, transA, transB,
                        m, n, k,
                        &alpha,
                        (void*)in_ptr, 
                        (void*)weight_ptr, 
                        &beta,
                        (void*)out_ptr);
    }
    
    return true;
}

