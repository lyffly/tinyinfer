#include <complex>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// modify from https://github.com/karpathy/llm.c/blob/master/dev/cuda/gelu_forward.cu

template<typename T>
__global__ void gelu_forward_kernel1(const T* inp, T* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = T(inp[i]);
        float cube = 0.044715f * xi * xi * xi;
        out[i] = T(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}


bool gelu_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape,
        std::string dtype, std::string layout, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int length = 1;
    for (auto& shape :in_shape) {
        length *= shape;
    }

    int block_size = 512;
    int grid_size = ceil_div(length, block_size);
    
    if (dtype=="float32") {
        gelu_forward_kernel1<float><<<grid_size, block_size, 0, stream>>>((float*)in_ptr, (float*)out_ptr, length);
    } else if (dtype=="float16") {
        gelu_forward_kernel1<half><<<grid_size, block_size, 0, stream>>>((half*)in_ptr, (half*)out_ptr, length);
    } else {
        printf("Gelu not support !!! \n");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
