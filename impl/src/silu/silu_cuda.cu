#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"


__global__ void silu_cuda_kernel_fp16(const size_t count, const half *input, half *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = __half2float(input[index]);
    float out_val = val / (1 + __expf(-val));
    output[index] = __float2half_rn(out_val);
}

__global__ void silu_cuda_kernel_fp16_half2(const size_t count, const half2 *input, half2 *output) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count) return;
        half2 h_val = input[index];
        float2 f_val = __half22float2(h_val);

        half2 t_val;
        t_val.x = __float2half_rn(f_val.x / (1 + __expf(-f_val.x)));
        t_val.y = __float2half_rn(f_val.y / (1 + __expf(-f_val.y)));
        output[index] = t_val;
}

__global__ void silu_cuda_kernel_fp32(const size_t count, const float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = input[index];
    float out_val = val / (1 + __expf(-val));
    output[index] = out_val;
}

__global__ void silu_cuda_kernel_fp32_float2(const size_t count, const float2 *input, float2 *output) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count) return;
        float2 h_val = input[index];

        float2 t_val;
        t_val.x = h_val.x / (1 + __expf(-h_val.x));
        t_val.y = h_val.y / (1 + __expf(-h_val.y));
        output[index] = t_val;
}


bool silu_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape,
        std::string dtype, std::string layout, int64_t pstream) {
    
    cudaStream_t stream = (cudaStream_t)pstream;
    int batch = out_shape.at(0);
    size_t length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }
    size_t block_size = 512;
    size_t grid_size = (length + block_size - 1) / block_size;
    if (dtype=="float32" ) {
        if (length % 2 == 0) {
            grid_size = (length/2 + block_size - 1) / block_size;
            silu_cuda_kernel_fp32_float2<<<grid_size, block_size, 0, stream>>>(
                length / 2, (const float2*)in_ptr, (float2*)out_ptr);   
        } else {
            silu_cuda_kernel_fp32<<<grid_size, block_size, 0, stream>>>(
                length, (const float*)in_ptr, (float*)out_ptr);   
        }
    } else if ( dtype=="float16" ) {
        if (length % 2 == 0) {
            grid_size = (length/2 + block_size - 1) / block_size;
            silu_cuda_kernel_fp16_half2<<<grid_size, block_size, 0, stream>>>(
                length / 2, (const half2*)in_ptr, (half2*)out_ptr);   
        } else {
            silu_cuda_kernel_fp16<<<grid_size, block_size, 0, stream>>>(
                length, (const half*)in_ptr, (half*)out_ptr);   
        }
    } else {
        printf("silu not support !!! \n");
    }
    return true;
}
