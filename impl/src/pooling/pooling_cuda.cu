#include <cudnn_v9.h>
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

__device__ inline float numerical_min(float a){
    return -1000.f;
}
__device__ inline __half numerical_min(__half a){
    return __float2half(-1000.f);
}

// from ppl.kernel.cuda/blob/master/src/nn/pooling_max.cu
template <int TILE_H, int TILE_W, typename T>
__global__ void ppl_cukernel_pooling_max_common(
    const T* input,
    T* output,
    int batch,
    int pad_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (c >= pad_channels) return;

    int inOff = (b * pad_channels + c) * in_height * in_width;
    int outOff = (b * pad_channels + c) * out_height * out_width;

    int partW = (out_width + TILE_W - 1) / TILE_W;

    int ox = (tx % partW) * TILE_W;
    int oy = (tx / partW) * TILE_H;

    // register blocking for input
    for (int i = 0; i < TILE_H; i++) {
        for (int j = 0; j < TILE_W; j++) {

            T res = numerical_min(T(0));

            // read input
            for (int fy = 0; fy < kernel_height; fy++) {
                for (int fx = 0; fx < kernel_width; fx++) {
                int iy = (oy + i) * stride_height + fy - pad_height;
                int ix = (ox + j) * stride_width + fx - pad_width;
                bool pred = (iy >= 0 && iy < in_height) && (ix >= 0 && ix < in_width);
                T ival = pred ? input[inOff + iy * in_width + ix] : numerical_min(T(0));

                res = (res > ival) ? res : ival;
                }
            }

            // store output
            if (oy + i < out_height && ox + j < out_width) {
                output[outOff + (oy + i) * out_width + ox + j] = res;
            }
        }
    }
}


bool pooling_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> kernels,
                     std::vector<int> paddings, std::vector<int> strides, std::vector<int> in_shape,
                     std::vector<int> out_shape, std::string optype, std::string dtype,
                     std::string layout, int64_t pstream, void* desc) {
    // PoolDesc* desc_ = (PoolDesc*)desc;

    if(optype == "MaxPool"){
        int batch = in_shape.at(0);
        int inc = in_shape.at(1);
        int inh = in_shape.at(2);
        int inw = in_shape.at(3);

        int outc = out_shape.at(1);
        int outh = out_shape.at(2);
        int outw = out_shape.at(3);

        int partH = (outh + 3) / 4;
        int partW = (outw + 0) / 1;
        dim3 dim_block(32, 4, 1);
        dim3 dim_grid;
        dim_grid.x = (partH * partW + dim_block.x - 1) / dim_block.x;
        dim_grid.y = (inc + dim_block.y - 1) / dim_block.y;
        dim_grid.z = batch;
        if (dtype=="float32") {
            ppl_cukernel_pooling_max_common<4, 1, float><<<dim_grid, dim_block, 0, (cudaStream_t)pstream>>>(
                    (float*)in_ptr, (float*)out_ptr, batch, inc, inh, inw, outh,
                    outw, kernels.at(0), kernels.at(1), strides.at(0), strides.at(1),
                    paddings.at(0), paddings.at(1));
        } else {
            ppl_cukernel_pooling_max_common<4, 1, __half><<<dim_grid, dim_block, 0, (cudaStream_t)pstream>>>(
                    (__half*)in_ptr, (__half*)out_ptr, batch, inc, inh, inw, outh,
                    outw, kernels.at(0), kernels.at(1), strides.at(0), strides.at(1),
                    paddings.at(0), paddings.at(1));
        }
        return true;
    }

    return true;
}
