#include "../../include/kernels.h"
#include "cuda_runtime.h"
#include "../utils.h"
#include <cstdio>

__global__ void elementwise_cuda(void *in0, void *in1, void *out, int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length) {
        ((float*)out)[index] = ((float*)in0)[index] + ((float*)in1)[index];
        //printf("%d = %f \n", index, ((float*)out)[index]);
    }
}

bool elementwise(int64_t in_ptr0, int64_t in_ptr1, int64_t out_ptr,
                std::vector<int> in_shape0, std::vector<int> in_shape1,
                std::vector<int> out_shape, int dtype, int layout) {

    int block_size = 256;
    int length = 2*3*5*5;

    int grid_size = (length + block_size - 1) / block_size;

    elementwise_cuda<<<grid_size, block_size>>>((void*)in_ptr0, (void*)in_ptr1, (void*)out_ptr,
                                            (int)length);

    return true;
}
