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

enum ActivationOpType {
    Activation_Unknown = 0,
    Activation_Relu,
    Activation_Silu,
    Activation_Elu,
    Activation_Selu,
    Activation_Sigmoid,
    Activation_LeakyRelu,
};

template <ActivationOpType op_type, typename T>
__device__ inline T activation_op(T a, float alpha, float beta);

template <>
__device__ inline float activation_op<Activation_Relu, float>(float a, float alpha, float beta) {
    return (a > 0) ? a : 0;
}
template <>
__device__ inline float activation_op<Activation_Sigmoid, float>(float a, float alpha, float beta) {
    return 1.f / (1.f + expf(-a));
}
template <>
__device__ inline float activation_op<Activation_LeakyRelu, float>(float a, float alpha,
                                                                   float beta) {
    return (a > 0) ? a : alpha * a;
}

template <>
__device__ inline half activation_op<Activation_Relu, half>(half a, float alpha, float beta) {
    return __hgt(a, 0) ? a : half(0);
}
template <>
__device__ inline half activation_op<Activation_Sigmoid, half>(half a, float alpha, float beta) {
    float in_valf = __half2float(a);
    float resf = 1.f / (1.f + expf(-in_valf));
    return __float2half(resf);
}
template <>
__device__ inline half activation_op<Activation_LeakyRelu, half>(half a, float alpha, float beta) {
    half res;
    res = __hgt(a, 0) ? a : __hmul((half)alpha, a);
    return res;
}

template <ActivationOpType op_type, typename T>
__global__ void activation_fp_cuda(T* in, T* out, int length, float alpha, float beta) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto in_ptr = reinterpret_cast<T*>(in);
    auto out_ptr = reinterpret_cast<T*>(out);
    if (index < length) {
        out_ptr[index] = activation_op<op_type, T>(in_ptr[index], alpha, beta);
    }
}

ActivationOpType string_to_activation_type(std::string op_type) {
    if (op_type == "Relu") {
        return Activation_Relu;
    } else if (op_type == "Elu") {
        return Activation_Elu;
    } else if (op_type == "Selu") {
        return Activation_Selu;
    } else if (op_type == "Sigmoid") {
        return Activation_Sigmoid;
    } else if (op_type == "LeakyRelu") {
        return Activation_LeakyRelu;
    } else {
        printf("Activation type not support.. \n");
        return Activation_Unknown;
    }
}

bool activation_backend(int64_t in_ptr, int64_t out_ptr, float alpha, float beta,
                        std::vector<int> in_shape, std::vector<int> out_shape, std::string dtype,
                        std::string layout, std::string optype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int block_size = 512;
    int length = 1;
    for (auto& shape : out_shape) {
        length *= shape;
    }

    int grid_size = (length + block_size - 1) / block_size;
    ActivationOpType act_op_type = string_to_activation_type(optype);
    if (optype == "Relu" && dtype == "float32") {
        activation_fp_cuda<Activation_Relu, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr, (float*)out_ptr, (int)length, alpha, beta);
    } else if (optype == "Relu" && dtype == "float16") {
        activation_fp_cuda<Activation_Relu, half><<<grid_size, block_size, 0, stream>>>(
            (half*)in_ptr, (half*)out_ptr, (int)length, alpha, beta);
    } else if (optype == "LeakyRelu" && dtype == "float32") {
        activation_fp_cuda<Activation_LeakyRelu, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr, (float*)out_ptr, (int)length, alpha, beta);
    } else if (optype == "LeakyRelu" && dtype == "float16") {
        activation_fp_cuda<Activation_LeakyRelu, half><<<grid_size, block_size, 0, stream>>>(
            (half*)in_ptr, (half*)out_ptr, (int)length, alpha, beta);
    } else if (optype == "Sigmoid" && dtype == "float32") {
        activation_fp_cuda<Activation_Sigmoid, float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr, (float*)out_ptr, (int)length, alpha, beta);
    } else if (optype == "Sigmoid" && dtype == "float16") {
        activation_fp_cuda<Activation_Sigmoid, half><<<grid_size, block_size, 0, stream>>>(
            (half*)in_ptr, (half*)out_ptr, (int)length, alpha, beta);
    }

    return true;
}
