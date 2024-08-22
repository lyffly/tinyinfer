#include <complex>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__forceinline__ __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__ float warpReduceSum(float sum, unsigned int threadNum) {
    if (threadNum >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (threadNum >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (threadNum >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (threadNum >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (threadNum >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

// modify from https://github.com/Bruce-Lee-LY/cuda_hgemv/blob/master/src/warp/warp1_naive.cu
// m=1  1*K @ K*N = 1*N

template<typename  T>
__global__ void gemv_nt_normal_kernel(const T* A, const T* B, T* C, const float* bias, float alpha, float beta, size_t N, size_t K) {
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (warp_col >= N) {
        return;
    }

    const size_t K_iters = div_ceil(K, WARP_SIZE);
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        const size_t A_idx = i * WARP_SIZE + lane_id;
        const size_t B_idx = i * WARP_SIZE + lane_id + warp_col * K;
        tmp += float(A[A_idx]) * float(B[B_idx]);
    }

    tmp = warpReduceSum(tmp, WARP_SIZE);

    if (lane_id == 0 && bias) {
        C[warp_col] = T(alpha * tmp + beta * bias[warp_col]);
    } else {
        C[warp_col] = T(alpha * tmp);
    }
}

bool gemv_cuda_backend(int64_t in_ptr, int64_t weight_ptr, int64_t bias_ptr, int64_t out_ptr,
                         int64_t workspace_size, int64_t workspace_ptr, float alpha, float beta,
                         bool transA, bool transB, std::vector<int> in_shape,
                         std::vector<int> weight_shape, std::vector<int> bias_shape,
                         std::vector<int> out_shape, std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int m = in_shape.at(0);
    int k = in_shape.at(1);
    int n = 0;

    if (transB) {    // n*k
        n = weight_shape.at(0);
    } else {    // k*n
        n = weight_shape.at(1);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(n, WARPS_PER_BLOCK));
    if (m == 1 && dtype=="float32" && transB) {
        gemv_nt_normal_kernel<float><<<grid, block>>>((float*)in_ptr, (float*)weight_ptr, (float*)out_ptr, (float*)bias_ptr, alpha, beta, n, k);
    } else if (m == 1 && dtype=="float16" && transB) {
        gemv_nt_normal_kernel<half><<<grid, block>>>((half*)in_ptr, (half*)weight_ptr, (half*)out_ptr, (float*)bias_ptr, alpha, beta, n, k);
    } else {
        printf("Gemv not support !!! \n");
    }
    return true;
}