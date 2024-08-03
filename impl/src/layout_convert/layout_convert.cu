#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"

enum LayoutConvertType {
    LayoutConvertType_Unknown = 0,
    LayoutConvertType_NCHW_2_NHWC,
    LayoutConvertType_NHWC_2_NCHW,
};

template <typename T>
__global__ void layout_convert_nchw2nhwc_fp_cuda(T* in, T* out, int N, int C, int H, int W,
                                                 int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto in_ptr = reinterpret_cast<T*>(in);
    auto out_ptr = reinterpret_cast<T*>(out);
    int n_i = index / (C * H * W);
    int c_i = index % (C * H * W) / (H * W);
    int h_i = index % (H * W) / W;
    int w_i = index % W;
    int index_out = n_i * (H * W * C) + h_i * (W * C) + w_i * C + c_i;
    if (index < length) {
        out_ptr[index_out] = in_ptr[index];
    }
}

template <typename T>
__global__ void layout_convert_nhwc2nchw_fp_cuda(T* in, T* out, int N, int C, int H, int W,
                                                 int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    auto in_ptr = reinterpret_cast<T*>(in);
    auto out_ptr = reinterpret_cast<T*>(out);
    int n_i = index / (H * W * C);
    int h_i = index % (H * W * C) / (W * C);
    int w_i = index % (W * C) / C;
    int c_i = index % C;
    int index_out = n_i * (C * W * W) + c_i * (H * W) + h_i * W + w_i;
    if (index < length) {
        out_ptr[index_out] = in_ptr[index];
    }
}

bool layout_convert_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                            std::vector<int> out_shape, std::string dtype, std::string in_layout,
                            std::string out_layout, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int batch = in_shape.at(0);
    int c = in_shape.at(1);
    int h = in_shape.at(2);
    int w = in_shape.at(3);
    // printf(" n c h w %d %d %d %d \n", batch, c, h, w);
    // printf(" ptr %p %p \n", (void*)in_ptr, (void*)out_ptr);
    int block_size = 512;
    int length = batch * c * h * w;
    int grid_size = (length + block_size - 1) / block_size;
    // printf(" length %d grid %d block %d \n", length, grid_size, block_size);
    // printf(" stream %p \n", stream);

    if (in_layout == "nhwc" && out_layout == "nchw" && dtype == "float32") {
        layout_convert_nhwc2nchw_fp_cuda<float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr, (float*)out_ptr, batch, c, h, w, (int)length);
    } else if (in_layout == "nhwc" && out_layout == "nchw" && dtype == "float16") {
        layout_convert_nhwc2nchw_fp_cuda<half><<<grid_size, block_size, 0, stream>>>(
            (half*)in_ptr, (half*)out_ptr, batch, c, h, w, (int)length);
    } else if (in_layout == "nchw" && out_layout == "nhwc" && dtype == "float32") {
        layout_convert_nchw2nhwc_fp_cuda<float><<<grid_size, block_size, 0, stream>>>(
            (float*)in_ptr, (float*)out_ptr, batch, c, h, w, (int)length);
    } else if (in_layout == "nchw" && out_layout == "nhwc" && dtype == "float16") {
        layout_convert_nchw2nhwc_fp_cuda<half><<<grid_size, block_size, 0, stream>>>(
            (half*)in_ptr, (half*)out_ptr, batch, c, h, w, (int)length);
    } else {
        std::cout << "[Error] layout convert cannot support \n";
    }
    return true;
}
