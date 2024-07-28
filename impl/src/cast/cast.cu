#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include "math.h"
#include <iostream>

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>

enum CastOpType {
    Cast_Unknown = 0,
    Cast_FP32_FP16,
    Cast_FP16_FP32,
    Cast_FP32_INT8,
    Cast_INT8_FP32,
};

template<typename T1, typename T2>
__device__ inline T2 cast_op(T1 input);


template<> __device__ inline float cast_op<half, float>(half input) {
    return __float2half(input);
}
template<> __device__ inline half cast_op<float, half>(float input) {
    return __half2float(input);
}

template<typename T1, typename T2>
__global__ void cast_fp_cuda(T1 *in, T2 *out, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto in_ptr = reinterpret_cast<T1*>(in);
    auto out_ptr = reinterpret_cast<T2*>(out);
    if (index < length) {
        out_ptr[index] = cast_op<T1, T2>(in_ptr[index]);
    }
}


bool cast_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape, std::vector<int> out_shape, 
                    std::string layout, std::string in_dtype, std::string out_dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int block_size = 512;
    int length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }
    // toto deal with broad cast and layout
    int grid_size = (length + block_size - 1) / block_size;

    if (in_dtype == "float16" and out_dtype == "float32") {
        cast_fp_cuda<half, float><<<grid_size, block_size,0, stream>>>((half*)in_ptr, (float*)out_ptr,
                                            (int)length);
    } else if (in_dtype == "float32" and out_dtype == "float16"){
        cast_fp_cuda<float, half><<<grid_size, block_size,0, stream>>>((float*)in_ptr, (__half*)out_ptr,
                                            (int)length);
    } 
    
    return true;
}
