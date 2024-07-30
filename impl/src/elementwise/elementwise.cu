#include <iostream>
#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

enum ElementwiseOpType {
    Elementwise_Unknown = 0,
    Elementwise_Add,
    Elementwise_Sub,
    Elementwise_Mul,
    Elementwise_Div,
    Elementwise_Max,
    Elementwise_Min,
    Elementwise_Pow,
};

template <ElementwiseOpType op_type, typename T>
__device__ inline T elementwise_op(T a, T b);

template <>
__device__ inline float elementwise_op<Elementwise_Add, float>(float a, float b) {
    return a + b;
}
template <>
__device__ inline float elementwise_op<Elementwise_Sub, float>(float a, float b) {
    return a - b;
}
template <>
__device__ inline float elementwise_op<Elementwise_Mul, float>(float a, float b) {
    return a * b;
}
template <>
__device__ inline float elementwise_op<Elementwise_Div, float>(float a, float b) {
    return a / b;
}
template <>
__device__ inline float elementwise_op<Elementwise_Max, float>(float a, float b) {
    return (a > b) ? a : b;
}
template <>
__device__ inline float elementwise_op<Elementwise_Min, float>(float a, float b) {
    return (a > b) ? b : a;
}
template <>
__device__ inline float elementwise_op<Elementwise_Pow, float>(float a, float b) {
    return powf(a, b);
}

template <>
__device__ inline int32_t elementwise_op<Elementwise_Add, int32_t>(int32_t a, int32_t b) {
    return a + b;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Sub, int32_t>(int32_t a, int32_t b) {
    return a - b;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Mul, int32_t>(int32_t a, int32_t b) {
    return a * b;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Div, int32_t>(int32_t a, int32_t b) {
    return a / b;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Max, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? a : b;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Min, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? b : a;
}
template <>
__device__ inline int32_t elementwise_op<Elementwise_Pow, int32_t>(int32_t a, int32_t b) {
    return powf(a, b);
}

template <>
__device__ inline half elementwise_op<Elementwise_Add, half>(half a, half b) {
    return __hadd(a, b);
}
template <>
__device__ inline half elementwise_op<Elementwise_Sub, half>(half a, half b) {
    return __hsub(a, b);
}
template <>
__device__ inline half elementwise_op<Elementwise_Mul, half>(half a, half b) {
    return __hmul(a, b);
}
template <>
__device__ inline half elementwise_op<Elementwise_Div, half>(half a, half b) {
    return __hdiv(a, b);
}
template <>
__device__ inline half elementwise_op<Elementwise_Max, half>(half a, half b) {
    return __hgt(a, b) ? a : b;
}
template <>
__device__ inline half elementwise_op<Elementwise_Min, half>(half a, half b) {
    return __hgt(a, b) ? b : a;
}
template <>
__device__ inline half elementwise_op<Elementwise_Pow, half>(half a, half b) {
    return __float2half(powf(__half2float(a), __half2float(b)));
}

template <ElementwiseOpType op_type, typename T>
__global__ void elementwise_fp_cuda(T* in0, T* in1, T* out, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto in0_ptr = reinterpret_cast<T*>(in0);
    auto in1_ptr = reinterpret_cast<T*>(in1);
    auto out_ptr = reinterpret_cast<T*>(out);
    if (index < length) {
        out_ptr[index] = elementwise_op<op_type, T>(in0_ptr[index], in1_ptr[index]);
    }
}

ElementwiseOpType string_to_elementwise_type(std::string op_type) {
    if (op_type == "Add") {
        return Elementwise_Add;
    } else if (op_type == "Sub") {
        return Elementwise_Sub;
    } else if (op_type == "Div") {
        return Elementwise_Div;
    } else if (op_type == "Mul") {
        return Elementwise_Mul;
    } else {
        printf("Elementwise type not support.. \n");
        return Elementwise_Unknown;
    }
}

bool elementwise_backend(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                         std::vector<int> in_shape0, std::vector<int> in_shape1,
                         std::vector<int> out_shape, std::string dtype, std::string layout,
                         std::string optype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int block_size = 512;
    int length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }
    // toto deal with broad cast and layout
    int grid_size = (length + block_size - 1) / block_size;
    ElementwiseOpType elementoptype = string_to_elementwise_type(optype);

    if (optype == "Add" && dtype == "float32") {
        elementwise_fp_cuda<Elementwise_Add, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr0, (float*)in_ptr1, (float*)out_ptr, (int)length);
    } else if (optype == "Add" && dtype == "float16") {
        elementwise_fp_cuda<Elementwise_Add, __half><<<grid_size, block_size, 0, stream>>>(
            (__half*)in_ptr0, (__half*)in_ptr1, (__half*)out_ptr, (int)length);
    } else if (optype == "Mul" && dtype == "float32") {
        elementwise_fp_cuda<Elementwise_Mul, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr0, (float*)in_ptr1, (float*)out_ptr, (int)length);
    } else if (optype == "Mul" && dtype == "float16") {
        elementwise_fp_cuda<Elementwise_Mul, __half><<<grid_size, block_size, 0, stream>>>(
            (__half*)in_ptr0, (__half*)in_ptr1, (__half*)out_ptr, (int)length);
    } else if (optype == "Sub" && dtype == "float32") {
        elementwise_fp_cuda<Elementwise_Sub, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr0, (float*)in_ptr1, (float*)out_ptr, (int)length);
    } else if (optype == "Sub" && dtype == "float16") {
        elementwise_fp_cuda<Elementwise_Sub, __half><<<grid_size, block_size, 0, stream>>>(
            (__half*)in_ptr0, (__half*)in_ptr1, (__half*)out_ptr, (int)length);
    } else if (optype == "Div" && dtype == "float32") {
        elementwise_fp_cuda<Elementwise_Div, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr0, (float*)in_ptr1, (float*)out_ptr, (int)length);
    } else if (optype == "Div" && dtype == "float16") {
        elementwise_fp_cuda<Elementwise_Div, __half><<<grid_size, block_size, 0, stream>>>(
            (__half*)in_ptr0, (__half*)in_ptr1, (__half*)out_ptr, (int)length);
    }
    return true;
}
