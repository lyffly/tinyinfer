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


// void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
//     const ggml_tensor * src0 = dst->src[0];
//     const float * src0_d = (const float *)src0->data;
//     float * dst_d = (float *)dst->data;
//     cudaStream_t stream = ctx.stream();

//     GGML_ASSERT(ggml_is_contiguous(src0));

//     GGML_ASSERT(src0->type == GGML_TYPE_F32);
//     GGML_ASSERT( dst->type == GGML_TYPE_F32);

//     const int64_t ne00 = src0->ne[0];
//     const int64_t nrows = ggml_nrows(src0);

//     float eps=1e-10;

//     rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
// }

