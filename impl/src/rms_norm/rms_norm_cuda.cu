#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include "../helper/helper.h"

#include "../../include/kernels.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"


// modify from llama.cpp/blob/master/ggml/src/ggml-cuda/norm.cu

template <int block_size, typename T>
__global__ void rms_norm_cuda_kernel(const T * x, T * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = float(x[row*ncols + col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = T(scale * float(x[row*ncols + col]));
    }
}


void rms_norm_fp32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_cuda_kernel<WARP_SIZE, float><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_cuda_kernel<1024, float><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

void rms_norm_fp16_cuda(const half * x, half * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_cuda_kernel<WARP_SIZE, half><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_cuda_kernel<1024, half><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}


bool rms_norm_cuda_backend(int64_t in_ptr, int64_t out_ptr, std::vector<int> in_shape,
                         std::vector<int> out_shape, float eps, std::string dtype, int64_t pstream) {
    cudaStream_t stream = (cudaStream_t)pstream;
    int batch = in_shape.at(0);
    size_t length = 1;
    for (auto& shape : in_shape) {
        length *= shape;
    }
    int nrows = length / batch;

    if ( dtype=="float32" ) {
        rms_norm_fp32_cuda((float*)in_ptr,  (float*)out_ptr,  batch, nrows,  eps,  stream);
    } else if ( dtype=="float16" ) {
        rms_norm_fp16_cuda((half*)in_ptr, (half*)out_ptr,  batch, nrows,  eps,  stream);
    } else {
        printf("rms norm not support !!! \n");
    }
    return true;
}
