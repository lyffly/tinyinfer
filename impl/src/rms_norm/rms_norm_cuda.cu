#include <math.h>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include <cub/cub.cuh>
#include <cub/util_type.cuh>
#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

// https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu


template <typename T>
__inline__ __device__ T _max(T a, T b) {
    return max(a, b);
}

template <typename T>
__inline__ __device__ T _sum(T a, T b) {
    return a + b;
}

#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)


template <typename T>
using ReduceFnType = T (*)(T, T);

// Helper function to return the next largest power of 2
__device__ __host__ static constexpr int _nextPow2(unsigned int num) {
    if (num <= 1)
        return num;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T, int numLanes = WARP_SIZE>
__inline__ __device__ T warpReduce(T val, ReduceFnType<T> fn) {
    static_assert(numLanes > 0 && (numLanes & (numLanes - 1)) == 0,
                  "numLanes is not a positive power of 2!");
    static_assert(numLanes <= WARP_SIZE);
#pragma unroll
    for (int mask = numLanes >> 1; mask > 0; mask >>= 1)
        val = fn(val, VLLM_SHFL_XOR_SYNC(val, mask));

    return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduce(T val, ReduceFnType<T> fn) {
    static_assert(maxBlockSize <= 1024);
    if constexpr (maxBlockSize > WARP_SIZE) {
        val = warpReduce<T>(val, fn);
        // Calculates max number of lanes that need to participate in the last
        // warpReduce
        constexpr int maxActiveLanes = (maxBlockSize + WARP_SIZE - 1) / WARP_SIZE;
        static __shared__ T shared[maxActiveLanes];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;
        if (lane == 0)
            shared[wid] = val;

        __syncthreads();

        val = (threadIdx.x < blockDim.x / float(WARP_SIZE)) ? shared[lane] : (T)(0.0f);
        val = warpReduce<T, _nextPow2(maxActiveLanes)>(val, fn);
    } else {
        // A single warpReduce is equal to blockReduce
        val = warpReduce<T, _nextPow2(maxBlockSize)>(val, fn);
    }
    return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceMax(T val) {
    return blockReduce<T, maxBlockSize>(val, _max<T>);
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceSum(T val) {
    return blockReduce<T, maxBlockSize>(val, _sum<T>);
}


template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t* input,     // [num_tokens, hidden_size]
                                const scalar_t* weight,    // [hidden_size]
                                scalar_t* out,             // [num_tokens, hidden_size]
                                const float epsilon, const int num_tokens, const int hidden_size) {
    __shared__ float s_variance;
    float variance = 0.0f;

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        const float x = (float)input[blockIdx.x * hidden_size + idx];
        variance += x * x;
    }
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = (float)input[blockIdx.x * hidden_size + idx];
        out[blockIdx.x * hidden_size + idx] = ((scalar_t)(x * s_variance)) * weight[idx];
    }
}


bool rms_norm_cuda_backend(int64_t in_ptr, int64_t w_ptr, int64_t out_ptr,
                           std::vector<int> in_shape, std::vector<int> out_shape, float eps,
                           std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int hidden_size = in_shape.back();
    size_t length = 1;
    for (auto& shape : in_shape) {
        length *= shape;
    }
    int num_tokens = length / hidden_size;
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));

    if (dtype == "float32") {
        rms_norm_kernel<float><<<grid, block, 0, stream>>>(
            (float*)in_ptr, (float*)w_ptr, (float*)out_ptr, eps, num_tokens, hidden_size);
    } else if (dtype == "float16") {
        rms_norm_kernel<half><<<grid, block, 0, stream>>>(
            (half*)in_ptr, (half*)w_ptr, (half*)out_ptr, eps, num_tokens, hidden_size);
    } else {
        printf("rms norm not support !!! \n");
    }
    return true;
}
